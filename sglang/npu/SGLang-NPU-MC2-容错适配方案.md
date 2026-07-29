# SGLang NPU + MC2 容错适配方案：拓扑变更不重捕图

生成时间：2026-07-29
目标硬件：华为昇腾 NPU
目标场景：DP-attention、ATTN_TP=1、moe_a2a_backend=deepep（MC2）、enable_dp_lm_head、纯 EP=DP
核心目标：**单个 DP rank 故障时，靠 MC2 的 mask 容错 + 固定通信组（不换 comm）继续推理，CUDA/NPU graph 不因容错而重捕；控制面走可重建的独立 gloo PG。**

> 本文区分：**原理闭环**（机制已验证可行）/ **工程待办**（方向明确但未实现）/ **待钉死**（最后一两个事实确认中）。

---

## 1. 目标约束与非目标

### 1.1 约束
- **只有 MoE 通信（dispatch/combine）使用 MC2 容错算子**（带 `x_active_mask` / `elastic_info` / active_ranks），其余集合通信不得引入同类容错后端。
- **`_TP`（全局 rank 大组）在 ATTN_TP=1 下不得参与任何跨 rank 集合通信**——所有跨 DP 通信要么消除、要么内化进 MC2 的独立 EP 组、要么走独立容错 gloo PG（仅控制面）。
- **不换通信组来容错**：容错靠 mask（数据面）和重建图外 gloo PG（控制面），不重建图内的 NCCL/HCCL comm。
- **不重捕图**：NPU graph（`torch.npu.NPUGraph`）固化的 comm 句柄在容错后保持有效。

### 1.2 非目标
- 不追求 ATTENTION_TP>1 / A>1 的 rejoin。
- 不实现 DeepEP/NixlEP（那是 GPU 路径），NPU 上用 MC2。
- 不处理 PD 分离、PP>1、speculative。

---

## 2. 核心原理：为什么这条路可行

### 2.1 关键洞察
"换通信组必重捕图"在 NPU 上成立（MC2 的 `group_ep`/`group_tp` 是**编译期固化进 NPUGraph 的 required string attr**，经 `CallHcomGetCommHandleByGroup` 解析成 HcclComm 句柄）。**所以容错绝不能靠换 group，必须靠"固定 group + mask 跳过死 rank"。**

MC2 的容错机制（`x_active_mask` + `elastic_info`）正是为此设计：group 不变（图内句柄有效 → 不重捕），死 rank 用 mask 在运行时跳过 → 容错。

### 2.2 与 vLLM 的对照根基
vLLM external-LB + FT 免重捕的三层架构，SGLang 可一一对应：

| 层 | vLLM | SGLang 适配 |
|---|---|---|
| 图内跨 rank 集合 | 无（只剩本地 GEMM） | 全部消除/内化（D1~D7 见 §3） |
| all2all 数据面 | DeepEP/NixlEP 独立资源 + mask | MC2 独立 EP 组 + active_ranks mask |
| 拓扑协商（图外） | stateless gloo cpu_group，可重建 | 独立容错 gloo PG，可重建（C1 见 §4） |

---

## 3. 数据面：推理阶段每一次通信的处理（D1~D7）

一次 decode forward 中所有跨 DP rank 集合通信的逐点处理方案。

| 点 | 名称 | 位置 | MC2 路径下的处理 | 图内？ | 组 | 状态 |
|---|---|---|---|---|---|---|
| **C1** | 控制面 MLP sync | dp_attn.py:158 | **独立容错 gloo PG**（§4） | 否（图外） | 独立 gloo PG | 工程待办 |
| **D1** | embedding 输入 gather | deepseek_v4.py:2326 | **不发生**（deepep 分支 `a2a.is_none()=False`，:2318 条件不满足） | — | — | ✅ 原理闭环 |
| **D2** | MoE 前 hidden 聚合 | communicator.py:1129 | **内化进 MC2 dispatch**（scatter-mode=SCATTERED，:385-399；dispatch 直接吃本 DP 分散 token，无独立 `_TP` 聚合） | 是（MC2 op 一部分） | `moe_ep_group.device_group` | ✅ 原理闭环 |
| **D3** | MoE 后散回 | communicator.py:1371 | **内化进 MC2 combine** | 是（MC2 op 一部分） | `moe_ep_group.device_group` | ✅ 原理闭环 |
| **D4** | logits hidden gather | logits_processor.py:785 | **开 `enable_dp_lm_head` 消除**（:355 → `do_tensor_parallel_all_gather_dp_attn=False`） | — | — | ✅ 原理闭环（配置） |
| **D5** | logits vocab all_gather | logits_processor.py:710 | **开 `enable_dp_lm_head` + ATTN_TP=1 消除**（:352-353 → `do_tensor_parallel_all_gather=False`） | — | — | ✅ 原理闭环（配置） |
| **D6** | logits scatter 回 local | logits_processor.py:830 | **纯本地 memcpy，无跨 rank**（dp_attention.py:649 `memcpy_triton`）；开 dp_lm_head 后连拷贝都省（:823 不触发） | 是（若发生） | 无 | ✅ 本就无通信 |
| **D7** | post-experts all_reduce | deepseek_v2.py:1009 / fused_moe_triton/layer.py:1396 | **不发生**——deepep 路径走 `forward_deepep`（:1196-1422），该分支结构性无 post-experts all_reduce；且 `reduce_results=False`（layer.py:212）。combine 已沿 EP 组内化规约 | — | — | ✅ 原理闭环（无需改动） |

### 3.1 D2/D3 内化的机制（已闭环）
`communicator.py:385-399` 的 scatter-mode 分派：
```python
if context.is_layer_sparse:
    if not get_moe_a2a_backend().is_none():   # deepep/fuseep → True
        return ScatterMode.SCATTERED   # prepare_mlp 不做 DP 聚合
    return ScatterMode.FULL            # none → 才做独立 _TP 聚合
```
MC2（deepep）路径选 `SCATTERED` → dispatch 直接吃本 DP 分散 token，聚合内化进 MC2 all2all，组为 `get_moe_ep_group().device_group`（**独立 EP 组，非 `_TP`**）。

### 3.2 D4/D5 消除的机制（配置即闭环）
`enable_dp_lm_head` 让每个 DP 用完整 W + 自己的 hidden **独立算完整 vocab 分布并本地出 token**（各 DP 请求独立，无需聚合）。代价是每 DP 存一份完整 LM head 权重（显存换通信），对正确性无影响。**这正是 mooncake Elastic EP 作者测试样例开启该参数的原因**——DP 弹性容错要求 logits 阶段不依赖全员在线的 `_TP` 大集合。

### 3.3 D6 本就无通信
`_scatter_dp_attn_logits` → `dp_scatter` 只做本地 `fill_(0)` + `memcpy_triton`，无 collective。

---

## 4. 控制面：C1 走独立容错 gloo PG

### 4.1 为什么 C1 必须独立
C1（`MLPSyncBatchInfo.all_gather`，`scheduler_components/dp_attn.py:158`）是**唯一与数据面解耦、每 step 必发的跨 DP 集合**，用于对齐 `num_tokens / can_cuda_graph / forward_mode`，从而固定 NPU graph 形状。它当前跑在 `_TP.cpu_group`，gloo 无 mask，死 rank 会挂起。

### 4.2 方案：独立容错 gloo PG + stateless 重建
对照 vLLM `gpu_worker_sentinel.py:120`（`get_dp_group().cpu_group = stateless_init_...`）：

1. **建独立 FT gloo PG**：distributed 初始化时新建一个 gloo PG（含全部 DP rank），与 `_TP` 解耦，注册进 ParallelState。
2. **C1 换组**：`prepare_mlp_sync_batch_raw` 的 group 从 `_TP.cpu_group` 换成 FT gloo PG（只换 group，不改语义）。改动点（subagent 已定位）：
   - `dp_attn.py:277-292` group/device 选择插新分支；
   - `dp_attn.py:169-178` active_ranks 来源从 `get_tp_group()` 换成 FT PG；
   - `dp_attn.py:140-141` slot 形状（dp×tp×cp）按 FT PG 重算。
3. **stateless 重建 + 紧凑重编号**：故障时销毁旧 gloo PG、用新 port + recompact rank（0..N-1，不能带洞）重建。SGLang 无 stateless group 实现，需照抄 vLLM `stateless_init_torch_distributed_process_group` + `original_to_new` + `update_parallel_config`。
4. **图外安全**：C1 在 NPU graph 外，重建 gloo PG **不触发重捕**。

### 4.3 备选（更彻底但改动大）
C1 不走 gloo 集合，改走 FT 的 ZMQ 控制通道（watchdog 上报 + TokenizerManager 派生 `effective_active` mask），用带租约的消息传递决定"哪些 DP 参与、各多少 token"。省去 gloo 重建，但要重写 token 对齐逻辑。

---

## 5. 待钉死项

### 5.1 D7 是否冗余可跳过（验证中）
- **假设**：MC2/deepep combine 已沿 EP 组内化跨 rank 求和（对照 vllm `output_is_reduced()==True`），D7 的 `_TP` all_reduce 冗余。
- **若是冗余**：给 SGLang 加 `output_is_reduced()` 判断——combine 后端为已内化规约的（MC2/deepep/mooncake）时跳过 D7。落点：`fused_moe_triton/layer.py:1396` 与 `deepseek_v2.py:1009` 两处。
- **vllm 证据**：`fused_moe.py:414-421` 注释"In mc2commimpl and alltoallcommimpl, we do not need to all-reduce the final outputs since the outputs are already aggregated"。

### 5.2 C1 在 NPU+MC2 的具体落点（验证中）
NPU 的 `cpu_group` backend、C1 是否每 step 必发、换组后 graph 形状决策一致性、NPU 是否已有 stateless 重建机制。

---

## 6. 与 vLLM 的完整对比

| 维度 | SGLang（本方案） | vLLM external-LB + FT / vllm-ascend |
|---|---|---|
| **图机制** | `torch.npu.NPUGraph`（复用 CUDA graph 抽象，叶子换 NPU 原语） | ACLGraph（同为 torch_npu 图抽象） |
| **dispatch/combine 定位** | 本方案：D2/D3 内化进 MC2 dispatch/combine | 收编进可插拔 `all2all_manager.dispatch/combine` |
| **D2/D3 数据面** | MC2 独立 EP 组 + active_ranks mask | DeepEP/NixlEP 独立资源 + mask |
| **D2/D3 是否入图** | 入图（MC2 op 一部分），但组固定不换 → 不重捕 | 入图，资源独立不换 → 不重捕 |
| **post-experts 规约（D7）** | 当前无条件 `_TP` all_reduce（**冗余**） | `output_is_reduced()` 判断，内化规约的后端跳过 |
| **logits（D4/D5）** | 开 `enable_dp_lm_head` 消除 | sequence-parallel/finalize TP=1 跳过（隐式等价） |
| **控制面（C1）** | 独立容错 gloo PG + stateless 重建（**待实现**） | stateless gloo cpu_group，`get_dp_group().cpu_group=新pg` |
| **stateless 机制** | **无，需照抄 vLLM 新写** | `stateless_init_torch_distributed_process_group` + `original_to_new` 重编号 |
| **容错触发** | MC2 mask（数据面）+ 重建 gloo PG（控制面） | `clean_mask()` + 重建 gloo cpu_group |
| **重捕图** | 不（组固定） | 不（资源独立+gloo 图外） |

### 6.1 SGLang 独有的优势
- **scatter-mode 机制**（`communicator.py:385-399`）天然实现 vllm-ascend 的 `should_skip_allreduce_across_dp_group` 效果：MC2 路径下 D2/D3 自动内化，无需显式开关。
- **mooncake dispatcher 已带 active_ranks**（`mooncake.py:234,274`），数据面 mask 容错已具备。

### 6.2 SGLang 落后需补的
- **D7 无 `output_is_reduced()` 判断**（冗余 all_reduce）。
- **无 stateless group / 紧凑重编号**（控制面重建机制缺失）。

---

## 7. 最小改动清单（工程待办汇总）

| # | 改动 | 落点 | 量 |
|---|---|---|---|
| 1 | 开 `enable_dp_lm_head` | server_args | 配置 |
| 2 | D7 加 `output_is_reduced()` 跳过 | fused_moe_triton/layer.py:1396、deepseek_v2.py:1009 | ~2 处 gate |
| 3 | 建独立容错 gloo PG + getter | parallel_state / distributed 初始化 | 新增 1 组 |
| 4 | C1 换组到 FT gloo PG | dp_attn.py:277-292、:169-178、:140-141 | 3 处 |
| 5 | stateless 重建 + 紧凑重编号 | 照抄 vLLM stateless 机制 | 新增机制 |
| 6 | （可选）D7 换 reduce_scatterv / 内化 | moe/utils.py | 视 5.1 结论 |

**结论**：原理上无残留障碍。数据面（D1~D7）已闭环（消除/内化/配置），唯一工程残留是 D7 的冗余 all_reduce gate；控制面（C1）需新建独立容错 gloo PG + stateless 重建机制。两者合起来，SGLang 在 NPU + MC2 下可实现"拓扑变更不重捕图"的 DP 粒度容错。
