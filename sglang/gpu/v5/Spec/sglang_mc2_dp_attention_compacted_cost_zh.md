# SGLang DP Attention + MC2 紧凑视图适配成本分析

> 日期：2026-07-02  
> 目标读者：SGLang FT、NPU/MC2、MoE/EP 适配开发工程师  
> 代码基线：`D:\Codex\repos\sglang`  
> 关联背景：`SGLang_MC2.md`、`sglang_fault_tolerance_design_v6_compacted_rank_arch_zh.md`、`sgl-project/sglang#8961`、`sgl-project/sglang#15771`

---

## 1. 目标

本文只回答一个收敛后的工程问题：

在官方建议的 DP attention 形态下，如果 MC2 不能演进出 Mooncake 等价的 active-rank 能力，而是采用“存活 rank 重新压缩成 compacted view，再重建数据面通信域”的方式，那么 SGLang 侧的最小适配成本是什么。

本文不讨论通用 TP/PP 拓扑，也不重新展开完整 v6 事务协议。判断口径限定为：

- 先用“通信组的 physical members/顺序是否变化，或旧 group 是否包含 dead rank”判断数据面通信域刷新影响面；
- 再单独列出不属于通信域重建、但必须随 compacted view 同步刷新的配置和 MoE expert 状态。

---

## 2. 前提与约束

### 2.1 目标启动形态

讨论的目标形态是：

```bash
--tp 4
--enable-dp-attention
--dp 4
--moe-a2a-backend mooncake   # NPU 目标中替换为 mc2
--moe-dense-tp-size 1
--enable-dp-lm-head
--enable-eplb
```

这里的 `DP=EP=TP=4` 必须按 SGLang DP attention 语义理解：

- `TP=4` 是当前 torch/model worker world 的大小；
- `DP=4` 是 DP attention 的 data-parallel 维度，不是 native DataParallelController 外层 4 个 replica；
- `EP=4` 由 MoE A2A backend 隐式得到，DeepEP/Mooncake 路径会把 `ep_size` 强制设为 `tp_size`；
- 总 model worker 数是 4，不是 `DP * TP = 16`。

关键代码逻辑如下。DeepEP/Mooncake 都会把 `ep_size` 改成 `tp_size`，因此启动命令中没有显式 `--ep-size` 也会得到 `EP=TP`：

```python
# D:\Codex\repos\sglang\python\sglang\srt\server_args.py
if self.moe_a2a_backend == "deepep":
    self.ep_size = self.tp_size

if self.moe_a2a_backend == "mooncake":
    self.ep_size = self.tp_size
```

DP attention 的 rank 公式决定了 `--tp 4 --dp 4` 时 attention TP 退化为 1：

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\dp_attention.py
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

代入目标参数：

```text
attn_dp_size = 4
attn_tp_size = 4 // 4 // 1 = 1
```

### 2.2 模型不做切分

本文结论依赖一个关键约束：模型权重不做 tensor/model 切分。

在该约束下，dense attention/dense MLP、MoE expert 内部都不依赖原始 4-way tensor shard。`tp_size=4` 更像承载 DP attention 和 EP 的全局运行时 rank world，而不是模型参数切分的 TP 维。

如果目标模型存在 dense TP、PP、CP 等切分，需要额外确认故障隔离边界。只要隔离完整 DP shard block，健康 block 内的 shard group 成员和切分度不变，模型切分不会自动增加通信域重建或权重重切成本；具体增量成本见 3.4。

### 2.3 shrink 语义

以 logical rank 1 故障为例：

```text
logical ranks:        0  1  2  3
active mask:          1  0  1  1
logical -> compacted: 0 -1  1  2
compacted -> logical: 0  2  3
```

故障后的执行面应从 4-rank view 切换到 3-rank compacted view：

```text
effective tp_size = 3
effective dp_size = 3
effective ep_size = 3
attn_tp_size      = 1
moe_tp_size       = 1
```

不能只把 `dp_size` 从 4 改成 3，而保留 `tp_size=4`。DP attention 公式要求：

```text
attn_tp_size = tp_size / dp_size / attn_cp_size
```

因此在 `attn_cp_size=1` 的目标形态下，`tp_size`、`dp_size`、`ep_size` 必须作为同一个 effective topology 一起刷新。

### 2.4 Mooncake EEP 的其他启动形态

本文约束的是“模型不做切分”的目标形态，不等于 Mooncake EEP 只能这样启动。仓内测试至少能看到两类会超出该约束的形态。

第一类是不启用 DP attention，只开启 Mooncake Elastic EP：

```python
# D:\Codex\repos\sglang\test\registered\ep\test_mooncake_ep_small.py
other_args=[
    "--tp", "4",
    "--elastic-ep-backend", "mooncake",
    "--moe-a2a-backend", "mooncake",
    "--moe-dense-tp-size", "1",
    "--enable-eplb",
    "--ep-num-redundant-experts", "72",
]
```

这时：

```text
enable_dp_attention = false
attn_dp_size = 1
attn_tp_size = tp_size / 1 / 1 = 4
ep_size      = tp_size = 4
moe_tp_size  = 4 / 4 / 1 = 1
```

也就是说，MoE expert 内部仍不切分，但 attention/dense TP 是 4，已经超出“模型不做切分”的约束。

第二类是 hybrid DP attention，例如 `--tp 4 --dp 2 --enable-dp-attention`：

```python
# D:\Codex\repos\sglang\test\registered\ep\test_mooncake_ep_small.py
extra_args = [
    "--enable-dp-attention",
    "--dp", "2",
]
```

这时：

```text
attn_dp_size = 2
attn_tp_size = 4 / 2 / 1 = 2
ep_size      = 4
moe_tp_size  = 1
```

该形态超出“模型不做切分”的基础约束，但不等于所有 2-rank attention TP group 都要重建。若故障时隔离完整 2-rank DP shard block，健康 `_ATTN_TP` 可以复用；新增成本主要是块级隔离和控制面协同。

### 2.5 模型存在切分时的目标方案

模型切分不改变 rank 空间的基本规则：Torch default process group 仍保留启动时的 global rank；压缩的是业务 DP/EP rank，以及成员发生变化的新数据面 group 的本地 rank。`torch.distributed.new_group(ranks=[4, 5])` 接收的是 global rank `[4,5]`，该 group 内部再把它们映射为 local rank `[0,1]`。

因此，不能再用“surviving group 非 singleton”直接推出“必须重建”。正确判据是：

```text
物理成员或成员顺序变化、或旧 group 包含故障 rank -> 刷新通信域
物理成员和顺序均不变                         -> 原通信域可复用
```

在 attention TP/CP 切分场景中，如果采用“故障 rank 所在整个 DP 副本一起隔离”的策略，健康 DP 副本内部的模型切分 group 成员不变，通常不需要重建，也不需要重切健康副本的权重。切分场景相对“不切分”的增量成本见 3.4。

---

## 3. 通信组影响面

### 3.1 初始 4-rank 形态

在：

```text
tp_size=4
dp_size=4
ep_size=4
attn_cp_size=1
moe_dp_size=1
pp_size=1
```

下，推导结果为：

```text
attn_dp_size = 4
attn_tp_size = 4 / 4 / 1 = 1
moe_ep_size  = 4
moe_dp_size  = 1
moe_tp_size  = 4 / 4 / 1 = 1
```

对应 GroupCoordinator：

| group | rank 数 | 当前语义 | 是否非平凡 |
|---|---:|---|---|
| `_TP` | 4 | 全局 runtime group，DP attention 和普通 collective 复用 | 是 |
| `_MOE_EP` | 4 | EP group，但 `ep_size == tp_size` 时直接别名 `_TP` | 是，但不单独计 |
| `_ATTN_TP` | 1 | attention tensor parallel group | 否 |
| `_ATTN_CP` | 1 | attention context parallel group | 否 |
| `_MOE_DP` | 1 | MoE data parallel group | 否 |
| `_MOE_TP` | 1 | MoE tensor parallel group | 否 |
| `_PP` | 1 | pipeline group | 否 |

对应的 group 构造代码可以简化成下面的伪代码。关键点是：`_MOE_EP` 在 `ep_size == tp_size` 时直接复用 `_TP`，而其他派生组的 size 由 `attn_tp_size`、`moe_tp_size`、`moe_dp_size` 决定：

```python
# D:\Codex\repos\sglang\python\sglang\srt\distributed\parallel_state.py
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size

if attn_tp_size == tensor_model_parallel_size:
    _ATTN_TP = _TP
else:
    _ATTN_TP = groups_of_size(attn_tp_size)

if moe_dp_size == tensor_model_parallel_size:
    _MOE_DP = _TP
else:
    _MOE_DP = groups_across_moe_dp_axis()

if moe_ep_size == tensor_model_parallel_size:
    _MOE_EP = _TP
else:
    _MOE_EP = groups_of_size(moe_ep_size)

if moe_tp_size == tensor_model_parallel_size:
    _MOE_TP = _TP
else:
    _MOE_TP = groups_of_size(moe_tp_size)
```

代入目标参数：

```text
attn_tp_size = 1
moe_dp_size  = 1
moe_ep_size  = 4
moe_tp_size  = 1

_ATTN_TP = singleton groups
_MOE_DP  = singleton groups
_MOE_EP  = _TP
_MOE_TP  = singleton groups
```

### 3.2 故障后 3-rank compacted 形态

rank 1 故障后，数据面重建后的核心 device group 是：

```text
_TP(new) logical members: 0, 2, 3
_TP(new) compacted ranks: 0, 1, 2
```

在模型不切分约束下，其他 group 仍为 singleton 或 `_TP` 的别名：

| group | shrink 后动作 |
|---|---|
| `_TP` | 必须重建，成员从 4 个变为 3 个 |
| `_MOE_EP` | 不单独重建；应继续指向新的 `_TP` |
| `_ATTN_TP` | 仍是 singleton |
| `_ATTN_CP` | 仍是 singleton |
| `_MOE_DP` | 仍是 singleton |
| `_MOE_TP` | 仍是 singleton |
| `_PP` | 仍是 singleton |

因此，当前形态应按“成员是否变化或是否包含 dead rank”判断数据面刷新影响面。该判断得到的结论是：

> 当前目标形态下，真正需要重建的 forward device GroupCoordinator 只有 `_TP`；`_MOE_EP` 只是 `_TP` 的使用身份，不应重复计算一套通信域成本。

### 3.3 `_ATTN_TP` 非 singleton 时的编号变化

如果目标形态放宽到 `attention_tp_size > 1`，需要区分 Torch global rank、group local rank 和 compacted 业务 DP rank。业务 DP rank 的变化不会自动改写前两者。

例子：

```text
old _ATTN_TP:
  DP0: [0, 1]
  DP1: [2, 3]
  DP2: [4, 5]
  DP3: [6, 7]

故障 DP1: [2, 3]

surviving logical groups:
  [0, 1], [4, 5], [6, 7]

default WORLD global ranks:
  仍为 [0, 1, 2, 3, 4, 5, 6, 7]

compacted business DP ranks:
  old DP0 -> new DP0
  old DP2 -> new DP1
  old DP3 -> new DP2

surviving _ATTN_TP global members:
  [0, 1], [4, 5], [6, 7]

每个 _ATTN_TP 的 group-local ranks:
  [0, 1]
```

`torch.distributed.new_group(ranks=[4, 5], ...)` 中的 `[4,5]` 是 default `WORLD` 的 global rank；返回的 process group 内，global rank 4 和 5 的 local rank 分别是 0 和 1。缩容后 `[4,5]` 的物理成员、顺序和 local rank 都没有变化，因此该 `_ATTN_TP` 可以直接复用。

所以，`_ATTN_TP` 的修正结论是：

> 在整 DP 隔离语义下，surviving `_ATTN_TP` 的 global members 和 group-local rank 均不变，不需要重建。需要刷新的只是业务 DP rank、DP size 以及依赖这些字段的调度和 batch metadata。

### 3.4 模型切分场景的额外成本

在整 DP 隔离、surviving DP 内模型切分度不变的前提下，模型切分不会新增一类必须重建的通信域。仍需刷新的核心 group 是成员发生变化的跨 DP `_TP`；`_MOE_EP == _TP` 时只刷新 alias，可选的 `_PDMUX_PREFILL_TP_GROUP` 若启用则与 `_TP` 同步刷新。健康 DP 内的 `_ATTN_TP`、`_ATTN_CP` 等 group 成员不变，可以复用。

例如 `tp_size=8, dp_size=4, attn_tp_size=2`，隔离 DP1 `[2,3]` 后：

| 对象 | 缩容后动作 |
|---|---|
| default `WORLD` | 保持 8-rank global namespace，不重建 |
| 跨 DP `_TP` | 成员从 `[0..7]` 变为 `[0,1,4,5,6,7]`，需通过 stateless PG、backend mask 或其他 dead-rank-safe 机制刷新 |
| surviving `_ATTN_TP` | 仍为 `[0,1]`、`[4,5]`、`[6,7]`，不重建 |
| `_MOE_EP` | 若 alias `_TP`，刷新 alias；不重复创建第二套 group |

相对模型不切分，新增成本主要在控制面，而不是通信域数量：

- 故障隔离粒度从单 rank 扩大为完整 DP shard block，损失容量为 `attn_tp_size * attn_cp_size` 个 rank；
- active-rank 映射必须理解 DP block，统一刷新 `dp_rank/dp_size`、请求归属和 KV cache 清理范围；
- 一个 DP 内多个 shard rank 需要共同 pause、drain、ack，故障事务参与者更多；
- 被隔离 block 上的 expert/冗余 expert 一并丢失，EPLB 的迁移量和存活 rank 容量校验成本更高；
- 需要审计缓存的 shard metadata、graph 和 communicator handle，但健康 shard group 的权重和通信域无需因业务 DP 编号变化而重切或重建。

只有当缩容策略改变了模型切分 group 的物理成员或 cardinality，例如保留半个 DP、降低 attention TP/CP 度、重组 `moe_tp_size > 1` 的 group、改变 PP stage 或在 `EP != TP` 下重新分组时，才会真正新增 `_ATTN_TP`、`_ATTN_CP`、`_MOE_TP`、`_PP` 等通信域重建和权重重切成本。

---

## 4. 4 个 rank 之间仍存在普通集合通信

“只有 `_TP` 需要重建”不等于“只有 MoE dispatch/combine 有通信成本”。SGLang 中 DP attention 和若干普通路径也复用 `_TP`。

典型路径：

| 路径 | 使用的通信 |
|---|---|
| scheduler DP attention 同步 | `torch.distributed.all_gather_into_tensor(..., group=tp_group.device_group/cpu_group)` |
| DP attention token gather | `_TP.all_gather_into_tensor` 或 `_TP.all_reduce` |
| DP attention reduce-scatter | `_TP.reduce_scatter_tensor` |
| layer communicator | `_TP.reduce_scatterv`、`_TP.all_gather_into_tensor` |
| 普通 TP wrapper | `tensor_model_parallel_all_reduce/all_gather/broadcast` 均从 `get_tp_group()` 取 group |

相关调用可以归纳成下面几类代码。它们说明 `_TP` 上的成本不只来自 MoE dispatch/combine。

```python
# D:\Codex\repos\sglang\python\sglang\srt\managers\scheduler_dp_attn_mixin.py
torch.distributed.all_gather_into_tensor(
    global_info_tensor.flatten(),
    local_info_tensor,
    group=tp_group.device_group,
)
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\dp_attention.py
if get_attention_tp_size() == 1:
    get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
else:
    get_attention_tp_group().reduce_scatter_tensor(...)
    get_tp_group().all_gather_into_tensor(...)
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\dp_attention.py
if get_tensor_model_parallel_world_size() == get_attention_dp_size():
    get_tp_group().reduce_scatter_tensor(output, input)
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\distributed\communication_op.py
def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(input_, dim=-1):
    return get_tp_group().all_gather(input_, dim)
```

所以 MC2 适配不能只替换 dispatch/combine。即使 MoE 数据面由 MC2 承接，普通集合通信也必须切到新的 compacted `_TP`。

---

## 5. 适配成本结论

在“模型不做切分”的约束下，当前成本可以明确收敛为三类。

### 5.1 `_TP` 刷新

这是核心数据面成本。

FT 控制面发现故障后，让所有存活 logical rank 基于 active mask 计算同一份 compacted view，并重新初始化 `_TP`：

```text
default WORLD global ranks: [0, 1, 2, 3]  # 保持不变
old _TP global members:     [0, 1, 2, 3]
new _TP global members:     [0, 2, 3]
new _TP local ranks:        [0, 1, 2]
logical -> business rank:   {0: 0, 2: 1, 3: 2}
```

刷新后：

- `_TP` 的 device group/cpu group 指向新通信域；
- `_MOE_EP` 继续指向新的 `_TP`；
- 旧 `_TP` 不再参与 forward 数据面；
- MC2 EP state 使用同一份 logical-to-compacted 映射。

这里不要求重建 Torch default `WORLD`。关键是不能通过一个会等待 dead rank 的普通 `new_group` 路径创建新 `_TP`；可选实现包括 backend active mask、vLLM 式 stateless process group，或者把 default `WORLD` 重建作为代价更高的兜底方案。该部分复杂度主要来自 HCCL/NPU communicator 的 dead-rank-safe 创建和生命周期，而不是 SGLang 上层 group 数量。

### 5.2 多层配置刷新

这是必须做、但范围可控的成本。

SGLang 当前在启动期把 `tp_size/dp_size/ep_size` 派生值缓存到多处。shrink 后如果只替换 `_TP`，而不刷新这些字段，会出现 shape、rank、batch metadata 与新 group world size 不一致。

至少需要刷新：

| 模块 | 需要刷新的内容 |
|---|---|
| runtime topology | effective `tp_size/dp_size/ep_size`、logical-to-compacted 映射 |
| Scheduler | `dp_size`、`tp_size`、`moe_ep_size`、`attn_tp_size`、`attn_dp_rank` |
| ModelRunner | `dp_size`、`tp_size`、`moe_ep_size`、`moe_ep_rank`、`tp_rank` 的执行面视图 |
| DP attention 全局状态 | `_ATTN_DP_SIZE`、`_ATTN_DP_RANK`、`_LOCAL_ATTN_DP_SIZE`、`_LOCAL_ATTN_DP_RANK` |
| scheduler DP sync | `global_num_tokens`、all-gather 输出 shape、active rank fallback |
| MC2/NPU MoE adapter | `elastic_info`、`log2phy`、rank id、EP group name/world size/rank id |

这些字段在启动期已经被复制到对象成员和模块级全局变量。只替换 `_TP` 而不刷新配置，会让 shape 和 rank 语义继续停留在旧 world size。

```python
# D:\Codex\repos\sglang\python\sglang\srt\managers\scheduler.py
self.tp_size = server_args.tp_size
self.moe_ep_size = server_args.ep_size
self.dp_size = server_args.dp_size
self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
    compute_dp_attention_world_info(..., self.tp_size, self.dp_size, ...)
)
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\model_executor\model_runner.py
self.tp_size = tp_size
self.moe_ep_size = moe_ep_size
self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
self.moe_dp_size = server_args.moe_dp_size
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\dp_attention.py
_ATTN_DP_SIZE = dp_size
_LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\managers\scheduler_dp_attn_mixin.py
global_info_tensor = torch.empty(
    (self.dp_size, self.tp_size * self.cp_size, 6),
    dtype=torch.int64,
    device=device,
)
```

这部分可以通过 FT 控制面经 ZMQ 下发 topology/config refresh 指令逐层刷新。只要刷新入口集中，影响范围是可控的。

### 5.3 MoE 专家刷新

这是基于现有 EPLB/expert placement 能力的成本，算法风险不大，但有明确前提。

需要做的事情：

- 根据 active logical ranks 重新计算 expert placement；
- 更新 `ExpertLocationMetadata`；
- 生成 MC2 所需的 expert/rank 映射；
- 确保 dispatch/combine、EPLB、expert metadata 使用同一份 compacted view。

MoE expert 的本地槽位和 expert location 也在启动或更新时绑定了 EP size。这里的风险不是通信组数量，而是 shrink 后 expert slot/权重能否承载新的 placement。

```python
# D:\Codex\repos\sglang\python\sglang\srt\eplb\expert_location.py
num_physical_experts = num_logical_experts + server_args.ep_num_redundant_experts
ep_size = server_args.ep_size
assert num_physical_experts % ep_size == 0
num_local_physical_experts = num_physical_experts // ep_size
```

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\moe\fused_moe_triton\layer.py
self.moe_ep_size = get_moe_expert_parallel_world_size()
assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
self._num_local_routed = self._num_global_routed // self.moe_ep_size
self.num_local_experts = self._num_local_routed + num_fused_shared_experts
```

## 6. compacted rank 模式下是否还需要 MC2 mask

不需要 MC2 提供 Mooncake 那种 active-rank mask 能力。

Mooncake active-rank 方案是在原 rank 空间里继续运行：

```text
原 group:      [0, 1, 2, 3]
active mask:  [1, 0, 1, 1]
执行语义:      backend 内部跳过 rank 1
```

这种模式要求 backend 理解“group 仍然是 4 个 rank，但其中一个 rank 不可达”。Mooncake 的普通 collective 和 EP dispatch/combine 都围绕这个语义设计。

compacted rank 方案不同。它在进入 MC2 之前已经把坏 rank 从数据面移除：

```text
default WORLD:    global ranks [0, 1, 2, 3]，保持不变
旧 logical group: [0, 1, 2, 3]
active mask:      [1, 0, 1, 1]
新 HCCL group:    global members [0, 2, 3]
MC2 world:        group-local ranks [0, 1, 2]
```

因此 MC2 不需要接收 `[1,0,1,1]` 这种带洞 mask，也不需要在一个包含 dead rank 的 4-rank group 里做跳过逻辑。MC2 只需要在新的 compacted world 上工作。

但这不等于“MoE 数据面没有任何 backend 需求”。如果目标仍选择 MC2 作为 MoE A2A backend，compacted 模式下依赖的是 MC2 的重新初始化和 dispatch/combine 能力，而不是 Mooncake 式 mask 能力：

| 能力 | 是否需要 | 说明 |
|---|---|---|
| Mooncake 式 active-rank mask | 不需要 | 不要求 MC2 在旧 4-rank group 内跳过 dead rank |
| 在新 group/world 上重新初始化 | 需要 | shrink 后 MC2 dispatch/combine 必须绑定 3-rank compacted world |
| logical expert 到 compacted rank 的映射 | 需要 | 可以叫 `elastic_info`、`log2phy` 或其他名字，本质是 rank/expert 映射 |
| dispatch/combine 不访问 inactive logical rank | 需要 | 由 SGLang 生成的新 group 和 MC2 参数共同保证 |
| 无重建、仅靠 mask 续推 | 不需要 | 这是 Mooncake 路径，不是 compacted 路径 |

可以用下面的伪代码区分两种模式：

```python
# Mooncake active-rank mode
group = old_group_0_1_2_3
active_mask = [1, 0, 1, 1]
mooncake.dispatch(tokens, active_mask=active_mask)

# compacted-rank mode
snapshot = compact(active_mask=[1, 0, 1, 1])
new_group = create_stateless_group(snapshot.active_global_ranks)  # [0, 2, 3]
moe_backend_state = moe_backend.reinit(
    group=new_group,
    rank=snapshot.logical_to_compacted[logical_rank],
    world_size=snapshot.compacted_world_size,
    expert_mapping=snapshot.expert_mapping,
)
moe_backend.dispatch(tokens, state=moe_backend_state)
```

结论：

> 在 compacted rank 适配模式下，MC2 的“mask 能力”不是必要条件；必要条件是 MoE dispatch/combine backend 能随新 compacted group 重新初始化，并消费 SGLang 生成的 rank/expert 映射。故障隔离和 dead-rank 排除由 SGLang 的 topology/group 重建承担，不由 backend mask 承担。

### 6.1 NPU DeepEP-Ascend dispatch/combine 的 compacted 形态

如果 NPU 目标采用 `D:\Codex\repos\sgl-kernel-npu` 中的 DeepEP-Ascend 路径，dispatch/combine 并不天然依赖 MC2 的 active-rank mask。它的入口是 PyTorch/HCCL process group，再把 HCCL group name、EP world size 和 EP rank id 传给 Ascend op。

Python `Buffer` 初始化时直接从传入的 HCCL group 取 rank、world size 和 HCCL comm name：

```python
# D:\Codex\repos\sgl-kernel-npu\python\deep_ep\deep_ep\buffer.py
self.group = group
self.rank = group.rank()
self.group_size = group.size()
backend = group._get_backend(torch.device("npu"))
moe_all_to_all_group_name = backend.get_hccl_comm_name(self.rank)
```

C++ low-latency dispatch/combine 再把这些参数传给 NPU op：

```cpp
// D:\Codex\repos\sgl-kernel-npu\csrc\deepep\deep_ep.cpp
EXEC_NPU_CMD(aclnnMoeLowLatencyDispatchV2,
    x,
    topk_idx,
    ...,
    hcom_ep_name,  // groupEp
    num_ranks,     // epWorldSize
    rank,          // epRankId
    num_experts,
    ...
)

EXEC_NPU_CMD(aclnnMoeLowLatencyCombineV2,
    expand_x,
    expert_ids,
    ...,
    hcom_ep_name,
    num_ranks,
    rank,
    num_experts,
    ...
)
```

仓内还存在纯 HCCL all-to-all 策略：

```python
# D:\Codex\repos\sgl-kernel-npu\python\deep_ep\deep_ep\strategies\low_latency_strategy.py
dist.all_to_all(output_list, input_list, group=group)
```

因此，MoE dispatch/combine 也可以按 compacted 数据面工作：

```text
old EP group: [0, 1, 2, 3]
rank 1 failed
new HCCL EP group: [0, 2, 3]
backend sees:
  epWorldSize = 3
  epRankId    = 0/1/2
  groupEp     = new HCCL group name
```

这条路径不需要 MC2 mask。真正新增的约束在 expert layout 和 kernel 支持范围上。

当前 NPU DeepEP-Ascend 多处假设专家能被 EP world size 均分：

```cpp
// D:\Codex\repos\sgl-kernel-npu\csrc\deepep\ops2\op_host\cam_moe_dispatch_normal_tiling.cc
localMoeExpertNum = moeExpertNum / epWorldSize;
OP_TILING_CHECK(moeExpertNum % epWorldSize != 0, ...);
```

默认 optimized config 也没有把 3-rank EP 当作一等配置：

```python
# D:\Codex\repos\sgl-kernel-npu\python\deep_ep\deep_ep\buffer.py
config_map = {
    2: Config(...),
    4: Config(...),
    8: Config(...),
    16: Config(...),
    ...
}
assert num_ranks in config_map
```

所以 `4EP -> 3EP` 在 NPU compacted 数据面上的关键成本不是 mask，而是：

- 重新生成 `expert -> compacted ep rank` 的布局；
- 保证 backend 看到的 active expert 数或物理 expert slot 能满足 `moeExpertNum % epWorldSize == 0`，或者扩展 kernel 支持非均匀 expert 分布；
- 为 3EP 等 shrink 后 world size 补齐 config/tiling 支持；
- 销毁旧 `Buffer`、旧 group name、旧 op handle，按新 HCCL group 重建 dispatch/combine runtime。

因此，对 NPU 场景应采用更准确的表述：

> dispatch/combine 不必依赖 MC2 active-rank mask；它可以在 compacted HCCL EP group 上重新初始化。剩余成本是专家布局重排、active expert 均分约束、EP world size 的 kernel/config 支持，以及旧 runtime handle 的重建。

### 6.2 如果走补齐 MC2 对标 Mooncake 的路线

前面的 compacted rank 方案不要求 MC2 具备 Mooncake active-rank mask 能力。另一条路线是反过来对标 Mooncake：保留旧 rank namespace，让 backend 在包含故障 rank 的 group 内通过 active mask 跳过不可达 rank。

这条路线的目标语义是：

```text
old group:     [0, 1, 2, 3]
active mask:   [1, 0, 1, 1]
backend view:  group size 仍为 4，但 rank 1 不参与数据面
```

SGLang 里 Mooncake 不是只覆盖 MoE dispatch/combine。`GroupCoordinator` 在创建普通 process group 时就把 active mask 传给 Mooncake backend：

```python
# D:\Codex\repos\sglang\python\sglang\srt\distributed\parallel_state.py
active_ranks = torch.ones(len(ranks), dtype=torch.int32, device=self.device)
active_ranks_cpu = torch.ones(len(ranks), dtype=torch.int32)

device_group = torch.distributed.new_group(
    ranks,
    backend="mooncake",
    pg_options=MooncakeBackendOptions(active_ranks, recovered_rank),
)
cpu_group = torch.distributed.new_group(
    ranks,
    backend="mooncake-cpu",
    pg_options=MooncakeBackendOptions(active_ranks_cpu, recovered_rank),
)

self.active_ranks = active_ranks
self.active_ranks_cpu = active_ranks_cpu
```

Mooncake EP dispatch/combine 又在每次执行时显式读取 `ElasticEPStateManager.active_ranks`：

```python
# D:\Codex\repos\sglang\python\sglang\srt\layers\moe\token_dispatcher\mooncake.py
packed_recv_hidden, packed_recv_count, self.handle, event, hook = buffer.dispatch(
    hidden_states,
    topk_ids,
    active_ranks,
    self.num_max_dispatch_tokens_per_rank,
    self.num_experts,
    timeout_us,
    use_fp8=use_fp8,
    async_finish=...,
    return_recv_hook=...,
)

combined_hidden_states, event, hook = buffer.combine(
    hidden_states,
    topk_ids,
    topk_weights,
    active_ranks,
    timeout_us,
    self.handle,
    async_finish=...,
    return_recv_hook=...,
)
```

故障 mask 更新后还会刷新 EP member：

```python
# D:\Codex\repos\sglang\python\sglang\srt\elastic_ep\elastic_ep.py
state.active_ranks.copy_(tensor)
state.sync_active_to_cpu()
EPBuffer._buffer.update_ep_member()
```

因此，若 MC2 要补齐到 Mooncake 等价能力，最小算子集合不是“dispatch/combine 两个算子”，而是下面四组能力。

| 能力组 | 必须补齐的接口/算子 | 原因 |
|---|---|---|
| active-rank process group | `new_group(..., active_ranks, recovered_rank)`、device/cpu active mask、mask runtime update、destroy/recover | SGLang 的普通 group 初始化、active rank 上报、故障恢复都假设 group 对象能携带 active mask |
| 普通集合通信 | `all_reduce`、`quant_all_reduce`、`all_gather_into_tensor`、`all_gather`、`reduce_scatter_tensor`、`broadcast`、`broadcast_tensor_dict`；按路径还可能包括 `gather`、`reduce_scatter`、`reduce_scatterv`、`all_gatherv`、`barrier`、`send/recv`、`fused_allreduce_rmsnorm` | DP attention、scheduler sync、dense TP wrapper、attention TP、MoE TP/EP wrapper 都通过 `GroupCoordinator` 走这些 collective |
| CPU/control group | `broadcast_object`、`broadcast_object_list`、`all_gather_object`，或等价的 CPU-safe active-rank 控制面 | Mooncake 当前有 `mooncake-cpu`；如果 MC2 只补 device group，旧 Gloo CPU group 一旦包含 dead rank 仍会阻塞 |
| MoE EP buffer | `get_ep_buffer_size_hint`、`Buffer(group, bytes)`、`dispatch(active_ranks, ...)`、`combine(active_ranks, handle, ...)`、`update_ep_member()`、event/hook/timeout/fp8 handle 语义 | Mooncake EEP 的核心数据面在这里，active mask 会影响收发 rank、handle 生命周期和首次执行超时策略 |

这些算子还需要满足两个语义要求。

第一，普通 collective 不能等待 inactive rank，也不能把 inactive rank 的旧数据计入结果。例如 `_TP.all_reduce` 在 mask `[1,0,1,1]` 下必须只在 active ranks 间得到正确 sum；`reduce_scatter_tensor`、`all_gather_into_tensor` 还要定义输出 shape 是按原 group size 还是 active count。Mooncake 路线倾向保留原 group size，但 SGLang 上层如果要把 `effective_tp_size` 刷成 3，又会和“原 group size 仍为 4”的输出 shape 发生冲突。因此这条路线通常还要配套一层 shape/topology 语义约定。

第二，MoE dispatch/combine 不能访问 inactive rank 上的 expert。要么 backend 根据 active mask 内部跳过并重映射 expert，要么 EPLB 提前把流量迁到存活 rank 的冗余 expert。无论哪种方式，都要保证 `active_ranks`、expert placement、dispatch handle、combine handle 是同一代状态。

对 NPU/MC2 还要额外考虑 HCCL 边界：如果 MC2 底层仍调用一个包含 dead rank 的 HCCL communicator，那么 active mask 本身不能避免 HCCL 等待故障 rank。MC2 必须在 HCCL 之上实现真正的 mask-aware collective，或者在 backend 内部动态创建只含 active ranks 的 HCCL subgroup。后一种实现本质上是把 compacted group 重建下沉到了 MC2 内部，并不是 Mooncake 式“旧 group + mask”。

因此，两条路线的工程成本差异可以概括为：

| 路线 | SGLang 侧成本 | MC2/backend 侧成本 | 风险判断 |
|---|---|---|---|
| compacted rank + HCCL/MC2 reinit | 重建 compacted group，刷新 topology/config/expert placement | 在新 group 上重新初始化 dispatch/combine，支持 shrink 后 EP world size | 成本主要在 SGLang FT 控制面和 NPU kernel/config 适配，语义直接 |
| MC2 补齐 Mooncake parity | 可以少做部分 group 重建，但仍要刷新上层 effective topology 或定义旧 size 输出语义 | 补齐 active-rank process group、device/cpu masked collective、MoE masked dispatch/combine、member update、handle/graph 恢复 | 成本高，且容易把故障语义分散到普通 collective、CPU 控制面和 MoE 数据面 |

结论：

> 若对标 Mooncake 走“补齐 MC2”的路线，完整成本至少包括 mask-aware process group、普通集合通信、CPU/control group、MoE EP dispatch/combine 与 member update。它不是只补两个 MoE 算子，而是实现一套 Mooncake 等价的 active-rank backend。与之相比，compacted rank 路线把故障隔离放在 SGLang topology/group 重建层，MC2 只需要面向新 compacted world 正常初始化和执行。

---

## 7. 与 vLLM 思路的关系

vLLM fault-tolerance 路径提供了三个关键参照：

1. vLLM 创建 TP group 时同样传入 default `WORLD` 的 global ranks，并不存在“vLLM 只传 group-local 0、1，而 SGLang 传 4、5”的差异。
2. scale down 压缩的是业务 `data_parallel_rank/data_parallel_size`；Torch default `WORLD` 的 global rank 和 world size 不随之改写。
3. vLLM 对成员变化的动态通信域使用 stateless process group；健康 TP group 的 physical members 和 local rank 不变，因此不重建。

`WORLD` 和 stateless 并不是二选一：`WORLD` 保留稳定的进程身份和启动期 group，stateless PG 承载故障后成员发生变化、且不能依赖 dead rank 参与创建的动态通信域。

SGLang 开启 DP attention 后，差异主要在拓扑：多个 DP attention rank 共用 `_TP`，所以缩容会改变 `_TP` 的成员；不开启 DP attention 时，每个 native DP replica 通常拥有独立 TP world，隔离整个 replica 后，健康 replica 内的 TP group 与 vLLM 一样可以保持不变。

因此，SGLang 侧的目标状态应明确区分 global rank 与 effective topology：

```text
default WORLD:
  global rank/world size 不变

提交新的 effective topology 和动态 group:
  effective_tp_size = active_count
  effective_dp_size = active_count
  effective_ep_size = active_count
  _TP global members = active global ranks
  _TP local ranks = [0 .. active_count-1]
  _MOE_EP = _TP
  attn_tp_size = 1
  moe_tp_size = 1
```

---

## 8. 明确结论

在给定“模型不做切分”的约束下，本轮分析结论已经明确：

1. “group 是否非 singleton”只能判断它是否存在通信成本，不能判断是否需要重建；重建判据是 physical members/顺序是否变化，或者旧 group 是否包含 dead rank。
2. 当前 DP attention + EP 形态下，成员发生变化、需要刷新的核心 forward device group 只有 `_TP`；可选 duplicate `_TP` 同步刷新。
3. `_MOE_EP` 是 `_TP` 的别名，不应重复计算一套通信域重建成本。
4. 除 dispatch/combine 外，4 个 rank 之间仍有 DP attention、scheduler sync、普通 TP wrapper 等集合通信，这些都通过 `_TP` 受影响。
5. compacted rank 模式不需要 MC2 提供 Mooncake 式 active-rank mask；若仍选择 MC2 作为 MoE backend，则需要 MC2 能在新 compacted group 上重新初始化，并消费 rank/expert 映射。
6. NPU DeepEP-Ascend dispatch/combine 可以按 compacted HCCL EP group 工作，不必依赖 MC2 mask；但要求 expert layout、`epWorldSize`、`epRankId`、HCCL group name 和 kernel/config 支持同步刷新。
7. 适配成本可收敛为三类：
   - `_TP` 刷新：FT 控制面让存活 rank 重新初始化 compacted 通信域，逻辑相对简单；
   - 多层配置刷新：通过 ZMQ/topology refresh 指令刷新 scheduler、model runner、DP attention 全局状态和 MoE backend 参数，范围可控；
   - MoE 专家刷新：基于已有 EPLB/expert placement 实现，主要风险在 expert slot/权重是否能承载 shrink 后的新 placement。
8. 如果模型存在切分且故障策略隔离整个 DP shard block，健康 `_ATTN_TP/_ATTN_CP` 的 global members 和 group-local rank 不变，不新增通信域重建；额外成本是块级隔离、映射/配置刷新、多 rank 故障事务、KV 清理和更大的 expert 恢复量。
9. 如果不走 compacted rank，而是要求 MC2 对标 Mooncake，则必须补齐 active-rank process group、普通 masked collective、CPU/control masked group、MoE EP `dispatch/combine/update_ep_member` 等能力。该路线成本显著高于“新 group 上重新初始化 MC2”。

最终判断：

> 在模型不切分、PP/CP/MoE-TP 均为 singleton 的目标形态下，MC2 适配的动态通信域刷新成本可以收敛到 `_TP`；default `WORLD` 无需重建。剩余主要是运行时 effective topology 刷新和专家元数据刷新。

如果模型存在切分，最终判断要改为：

> 在整 DP 隔离且 surviving DP 内切分度不变时，目标方案仍只刷新成员变化的跨 DP `_TP`；健康模型切分 group 不重建、不重切权重。模型切分新增的是隔离粒度、控制面协同和 expert/KV 恢复成本。只有缩容改变某个 shard group 的成员或 cardinality 时，才增加该 group 的 communicator 重建、shard metadata 刷新和权重重切成本。

如果选择补齐 MC2 对标 Mooncake，最终判断要改为：

> 需要实现的是 Mooncake 等价 active-rank backend，而不是单独的 MoE dispatch/combine 替换。最小闭环包括普通 process group mask 语义、所有可达 collective 的 dead-rank 跳过语义、CPU 控制面安全语义、MoE EP buffer 的 active mask/handle/member update 语义，以及 NPU/HCCL 边界上避免等待 dead rank 的实现。

---

## 9. 不适用范围

以下任一条件成立时，本文的低成本结论需要重新评估。关键不在于“存在模型切分”，而在于缩容是否改变 shard group 的成员或切分度：

- 故障后保留不完整 DP shard block，或降低 attention TP/CP cardinality；
- `moe_tp_size > 1` 的 group 跨越被隔离边界，或缩容后需要重新分组；
- PP stage 成员丢失，且不能通过隔离完整 pipeline replica 保持健康 `_PP` 不变；
- `EP != TP`，shrink 后需要改变 EP group 成员；
- native DataParallelController 外层 DP replica 与 DP attention 混用；
- 多节点 rendezvous、HCCL group 重建能力、device reset 行为未验证；
- expert 权重在 shrink 后无法由存活 rank 的 slot/冗余专家覆盖；
- 模型路径存在直接缓存旧 `ProcessGroup` 或裸用旧 `torch.distributed.WORLD` 的 forward collective。
- NPU DeepEP/Ascend op 不支持 shrink 后的 `epWorldSize`，或 active expert 数无法满足 backend 的均分约束。

这些情况会扩大通信域、权重 shard 和配置刷新影响面，不能再只按当前 `_TP` 单组模型估算。
