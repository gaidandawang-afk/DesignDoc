# NPU MC2 Compact Survivor Topology：rank、通信组与 expert 映射

> 适用实现：SGLang `codex/review-0814-npu-ft-minimal` @ `127f5a6711`
>
> 目的：精确说明 compacted rank 的实际改动面、成本和限制

## 1. 为什么需要单独说明 rank 空间

NPU 静态缩容后，系统同时存在稳定 original id 和紧凑 communication id。如果不在文档、日志和代码接口中明确区分，很容易出现以下错误：

- 把 compact group rank 回报成外部 DP id；
- 用 original rank 直接作为新 HCCL group 的 root/src/dst；
- 用 global scheduler mask 初始化以 DP rank 为 world 的通信对象；
- 计算 physical expert id 时仍乘原 EP world size；
- 把 inactive original slot 当成有效 gather 结果；
- 连续 shrink 时不同 generation 误加入同一个 store prefix。

当前实现通过“外部 original、专用 group compact、固定 tensor 携带双向映射”降低这类风险。

## 2. 四类标识

### 2.1 Original DP rank

启动时确定的稳定 `dp_rank ∈ [0, D)`。它用于：

- FT status 中的 engine id；
- `removed_dp_ranks`；
- 外部 `routed_dp_rank`；
- incident 和 expected mask；
- expert placement 的持久语义；
- 日志和 artifact。

缩容后 original id 不重编号。

### 2.2 Global scheduler rank

若每个 DP 有 `R` 个 scheduler rank，则：

```text
W = D × R
global_rank ∈ [0, W)
```

它用于进程存活、节点 heartbeat、DPC shutdown 和发送内部命令。whole-DP 操作必须把一个 DP id 展开到它的全部 global ranks。

GPU 和 NPU Manager 都已生成这种 whole-DP global-rank mask。NPU 核对提交的 compact communication object 仍用 `dp_rank/dp_size` 初始化，这是 `R > 1` 路径中的局部坐标错误；目标应直接改用 `tp_rank/tp_size`，无需把 mask 压回 DP 空间。

### 2.3 Compact survivor group rank

设存活 original global scheduler rank 有序集合：

```text
G = (g0, g1, ..., gK-1)
```

则：

```text
compact_rank(gi) = i
effective_world_size = K
```

它只在新建的 Gloo MLP-sync group 和 HCCL EPLB group 内使用，不暴露到 HTTP API。一个 DP 有多个 scheduler rank 时，其所有 survivor sibling 都分别占一个 compact group rank；HTTP 仍只暴露 original DP id。

### 2.4 Effective EP rank

当前 NPU MC2 实现把 survivor original EP rank 映射到连续 effective EP rank。其数值和 compact survivor rank 一致，但概念不同：

- compact group rank 是 process-group 地址；
- effective EP rank 是 MC2 kernel 和 physical expert id 使用的 topology 地址。

保持概念分离有助于未来出现“每 DP 多 EP/scheduler rank”时扩展映射。

## 3. 示例：4 路缩到 3 路

本例假设每个 DP 只有一个 scheduler rank，用于展示基础映射；`attn_tp>1` 见 3.1。

初始：

```text
original ranks: 0  1  2  3
active mask:    1  1  1  1
```

移除 original rank 1：

```text
active mask:              1  0  1  1
active_original_ranks:   [0,    2, 3]
original_to_effective:   [0, -1, 1, 2]
effective_to_original:   [0,  2, 3, -1]
```

新专用 group：

| Original rank | Group rank | 是否参加 Gloo/HCCL survivor group |
|---:|---:|---|
| 0 | 0 | 是 |
| 1 | - | 否 |
| 2 | 1 | 是 |
| 3 | 2 | 是 |

外部 status 仍显示 engine 2、engine 3，而不是 engine 1、engine 2。

### 3.1 DP2 × ATTN_TP2

设 `D=2`、`attn_cp=1`、`attn_tp=2`，物理 global rank 布局为：

```text
DP0: global ranks [0, 1]
DP1: global ranks [2, 3]
```

移除 DP1 后，Manager 已生成：

```text
active_global_rank_mask = [1, 1, 0, 0]
active_original_ranks   = [0, 1]
```

DP0 的两个 attention-TP sibling 都执行 rebuild，并分别以 `tp_rank=0/1` 加入 compact group rank 0/1。MLP-sync gather 的两个结果写入 `global_info_tensor.view(-1, info_width)` 的 global slots 0/1，对应 `[DP0, ATTN_TP0]` 和 `[DP0, ATTN_TP1]`。这正是当前提交需要补齐的两处坐标修改；group、membership generation、EPLB 和 MC2 elastic-info 不需要换方案。

## 4. `elastic_info` 精确布局

设 original EP size 为 `E0`，每个 original EP rank 的本地 physical expert 数为 `L`。tensor 长度固定为：

```text
4 + 2 × E0
```

布局：

```text
offset 0              has_shrunk
offset 1              effective_ep_size = K
offset 2              reserved = 0
offset 3              effective_physical_expert_count = K × L
offset 4 .. 4+E0-1    original_to_effective
offset 4+E0 .. end    effective_to_original
```

对于上例 `E0=4`、`L=8`：

```text
[1, 3, 0, 24,  0, -1, 1, 2,  0, 2, 3, -1]
```

该 tensor 在设备上创建一次，之后只用 `copy_` 原地更新。shape、dtype 和 data pointer 保持不变。

## 5. physical expert id 映射

假设 original rank `r` 上本地 expert `l` 的 physical id 为：

```text
p_original = r × L + l
```

缩容后：

```text
e = original_to_effective[r]
p_effective = e × L + l      if e >= 0
              -1             if e < 0
```

例如 `L=8`：

| Original rank/local | Original id | Effective rank | Compact id |
|---|---:|---:|---:|
| 0/3 | 3 | 0 | 3 |
| 1/3 | 11 | -1 | -1 |
| 2/3 | 19 | 1 | 11 |
| 3/3 | 27 | 2 | 19 |

这一步必须在 MC2 dispatch 使用 placement 前完成。否则 kernel 会把 original 2 的 expert 当作 compact rank 2，而实际 group rank 2 已对应 original 3。

## 6. 专用通信组

### 6.1 Gloo MLP-sync group

用途：在 survivor DP 间执行 CPU/Gloo gather。创建后 barrier 确认全部成员进入同一代际。

上层输出仍按 original DP 槽位组织：compact gather 得到的第 `i` 个结果写回 original slot `S[i]`。inactive slot 不应被当成有效输出。

### 6.2 HCCL EPLB group

用途：EPLB 的统计、分布广播、P2P expert migration。创建后执行一个 device all-reduce warmup，确认 communicator 可用。

所有 group API 的 root/src/dst 必须经过：

```text
original rank -> active_original_ranks.index(original rank)
```

inactive original rank 不得进入 group 操作。

### 6.3 未被重建的 group

实际分支没有替换 SGLang 全局 `_TP` 或所有 model-parallel process group。旧文档将 compacted 方案成本估计为“重建整个 `_TP`”过于宽泛。

当前真实成本更窄：

- 两个专用 process group；
- EPLB 上下文切换；
- MLP-sync gather/write-back；
- MC2 topology tensor 和 expert id 改写；
- device stop/restart。

但改动窄也带来验证责任：必须证明没有遗漏仍跨 dead DP 使用旧 group 的路径。

## 7. rendezvous 与 generation

每次 rebuild 生成：

```text
membership = bit string of active mask
generation = previous generation + 1
prefix = npu-ft/<membership>/<generation>
```

同时使用 membership 和 generation 的原因：

- 不同 survivor 集合不能共享 store key；
- 同一 survivor 集合在重复恢复/重试时也不能误读旧 key；
- 日志可以关联 topology 与代际。

当前代码在每个 scheduler 进程内递增 generation。它假设 survivor 对操作顺序有一致观察；若某 rank 跳过一次 rebuild，下一次 prefix 可能不一致并超时。Manager 的单事务和 survivor ack 是必要条件。

## 8. 连续缩容

对 4→3→2：

```text
gen 1: active [0,2,3]
  original_to_effective = [0,-1,1,2]

gen 2: active [0,3]
  original_to_effective = [0,-1,-1,1]
```

每一代都必须：

1. survivor 对同一 active mask 达成一致；
2. stop/restart device；
3. 创建新 Gloo/HCCL group；
4. 重新计算 expert placement；
5. 更新 elastic-info；
6. 不再使用上一代 group。

分支代码允许 generation 递增，但尚无 NPU 硬件连续缩容 artifact，因此不能仅凭数据结构宣称连续 shrink 已验证。

## 9. expert 容量与精度约束

缩容不会凭空增加 survivor 的物理 expert 槽位。`effective_physical_expert_count = K × L` 明确减小，EPLB 必须把逻辑 expert placement 调整到剩余容量。

需要验证：

- 模型冗余 expert 是否足以覆盖失效 rank 上的逻辑 expert；
- 每个 survivor 的容量约束不被破坏；
- `num_physical_experts` 能按 original EP size 正确计算本地 `L`；
- inactive physical id 全部变成 `-1`，不会静默路由到错误 expert；
- rebalance 后权重/format copy 正确；
- 推理精度与缩容前基线符合预设容差。

源码能证明映射公式，不能证明特定模型在特定缩容比例下具备足够冗余容量。

## 10. internal-format tensor

Ascend tensor 可能带有设备内部格式。expert migration 不能假设普通 `.copy_()`、CPU staging 或 contiguous layout 总能保持语义。

NPU 分支在硬件 utils 和 EPLB P2P 路径中增加 format-aware staged copy。文档不应把它简化成性能优化；错误处理会导致权重内容或 layout 损坏，即使 communicator 本身成功。

验证时至少要保留：

- 源/目标 tensor dtype、shape、format；
- 迁移前后 checksum 或可解释数值比较；
- 缩容后模型精度；
- HCCL P2P/collective 日志。

## 11. 当前拓扑约束

### 11.1 已明确约束

- whole-DP scale-down；
- survivor 数至少为 1；
- NPU + MC2 + EPLB；
- pipeline parallel size 仍受旧 FT gate 限制；
- static shrink，不声明在线 rejoin；
- current code path 面向 pause 策略。

### 11.2 `attn_tp>1` 的当前实现差距

规范 rank 空间已经确定为 global scheduler rank。当前 `NpuFTCommunication` 仍使用 `original_rank=dp_rank`、`original_world_size=dp_size`，而 `rebuild()` 枚举 global active mask；NPU MLP-sync 又把 gather 结果写到固定 TP0 槽位。这两处使核对提交只在 `attn_tp=1` 时自然成立。

目标修正是：

```text
dp_rank/dp_size -> tp_rank/tp_size
global_info_tensor[active_ranks, 0]
  -> global_info_tensor.view(-1, info_width)[active_ranks]
```

预计产品代码少于 10 行，不需要增加新 process-group 类型，也不需要改变 whole-DP mask。修正后仍需单元测试证明 global slot 顺序，并在 NPU 上验证 DP2×ATTN_TP2 的 shrink、MLP-sync、EPLB、精度和持续请求。`attn_cp>1` 保持独立待验证项。

## 12. 失败模式

| 失败 | 可观察结果 | 必须行为 |
|---|---|---|
| survivor active mask 不一致 | PrefixStore/group name 不一致，超时 | fail-stop，保持路由关闭 |
| original rank 不在 active mask | `.index()` 失败 | 说明 removed rank 错收命令；fail-stop |
| Gloo barrier 超时 | group 未完成 | 重新进入 FT pause/进程失败 |
| HCCL warmup 失败 | communicator 不可用 | 不更新 EPLB context，不开放路由 |
| MLP gather 超时 | `NpuFTMLPSyncInterrupted` | scheduler 进入 FT control loop |
| expert id 映射到 -1 | 原 expert 不在 survivor | EPLB 必须重新 placement，不能派发 |
| tensor 地址改变 | Graph 参数可能 stale | 单测应阻止；设备验证失败 |
| generation 漂移 | survivor 互相等待不同 prefix | 有界超时并 fail-stop |

## 13. 成本总结

| 成本维度 | 当前实际成本 |
|---|---|
| 代码改动面 | FT communication、EPLB、ElasticEP、dispatcher、scheduler、NPU copy |
| 每次 shrink 控制成本 | stop/restart + 两个 group 创建 + barrier/warmup |
| 内存 | 固定 `4+2E0` elastic tensor、group/context 元数据、固定 original 槽位 |
| 映射 | original↔compact、physical expert id compact |
| Graph 风险 | 地址稳定已设计，设备 replay 未证明 |
| 连续 shrink | generation 支持设计存在，硬件行为未验证 |
| rejoin | 需要逆向事务，当前未实现 |
| 运维 | 必须记录 original id、compact id、membership、generation 和 store endpoint |

相比“重建全部 `_TP`”，当前实现降低了全局并行状态变更；相比“固定所有 group 只传 mask”，它增加了 communicator 创建和 rank 映射。这个折中只有在 NPU 硬件 E2E 证明所有必要 collective 都被覆盖后才算闭环。
