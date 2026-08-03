# SGLang DP-only FT NPU 兼容适配方案：GPU/NPU 一套代码

生成时间：2026-08-03
目标硬件：NVIDIA GPU（Mooncake elastic EP）+ 华为昇腾 NPU（MC2 / DeepEP-Ascend）
代码基线：`D:\Codex\repos\sglang-dp-only-ft-revise`（`worktree-dp-only-ft-revise@aa88dae21`）
关联文档：

- `SGLang DP-only FT 架构设计与验证指南`（`D:\Codex\shared\2026-07-17\`，下称 FT 指南）
- `SGLang-NPU-MC2-容错适配方案.md`（mask 容错 + 不重捕图路线，下称 MC2 方案）
- `sglang_mc2_dp_attention_compacted_cost_zh.md`（compacted rank 成本分析）
- vllm 参照：`D:\Codex\repos\vllm`、`D:\Codex\repos\vllm-ascend`

> 本文只讨论**架构与实现思路层面的兼容适配点**。支持门控、CLI 参数、超时调参、测试矩阵等与架构无关的工程项不在此列。
>
> 核心立场：**一套代码同时兼容 GPU 和 NPU。GPU 路径零删除、零行为变更；NPU 特殊性全部通过能力声明与 hook 注入，不新建平行控制面。**

---

## 1. 背景：两种设备上的故障恢复为什么不同

### 1.1 GPU/mooncake 现状：原地续推模型

Mooncake elastic EP 的容错语义是 active-rank mask：group 不变、comm 不变、故障 rank 在数据面被跳过。代码路径：

| 环节 | 位置 | 行为 |
| --- | --- | --- |
| dispatch/combine 读 mask | `layers/moe/token_dispatcher/mooncake.py:204-227, 252-274` | 每次 forward 实时读 `ElasticEPStateManager.active_ranks` 传给 Mooncake Buffer，故障 rank 的 token 在 backend 层被跳过 |
| 故障当拍自愈 | `model_executor/model_runner.py:3607-3634` `_maybe_rebalance_after_rank_fault` | 检测 active_ranks 变化 → 立即 EPLB rebalance → **用同一个 forward_batch 重做 `_forward_raw`**。scheduler 与请求层无感 |
| membership 恢复 | `model_executor/model_runner.py:1690-1729` `maybe_recover_ep_ranks` | 每次 forward 末尾调用；恢复 WORLD/派生 group/MQ/EP member，广播 expert metadata，重置 active_ranks |
| 控制面上报 | scheduler 每次 forward 后发 `ActiveRanksOutput` | 投影成 DP mask，驱动 FT 状态机 |

结果：**只要故障不发生在请求所在 DP，running batch 原地继续，请求无任何感知**。这是 GPU 路径的核心资产，兼容工作必须完整保留它。

### 1.2 NPU 硬件约束：必须 stop/restart_device

NPU 上集合通信要求全员参与，单 rank 故障会污染所有幸存 rank 的 HCCL 上下文（挂起的 collective 无法靠 mask 清除），硬件层面只有设备级重置能恢复。参照 vllm/vllm-ascend 的既有实现：

| 时机 | 动作 | 位置 |
| --- | --- | --- |
| 故障检测后立即 pause | `torch_npu.npu.stop_device(device.index)`，**广播到所有幸存 engine 的全部 worker** | `vllm-ascend/vllm_ascend/worker/sentinel/npu_worker_sentinel.py:95-106` |
| retry/scale_down 恢复阶段 | `clean_states()`：`stop_device` + `restart_device` + `torch_npu.distributed.reinit_process_group` + 清空 input_batch/model state，同样在**所有幸存 worker** 上执行 | 同上 `:228-251` |
| in-flight 请求 | `fault_tolerant_wrapper` 把**所有** running 请求 `preempt_request()` 回 waiting 队列（释放 KV、重置计算进度），恢复后自动重算，客户端无感 | `vllm/v1/fault_tolerance/engine_core_sentinel.py:466-552` |

两个关键事实决定了兼容方案的形状：

1. **幸存 rank 也要 stop/restart**。这与 GPU/mooncake 形成根本分歧——GPU 上幸存者无需任何设备操作。
2. **vllm 不做续推**：任何故障都是 pause → 全量 preempt → stop/restart → 重建 PG → resume → 重算。

### 1.3 对 sglang 的直接推论

- NPU 上 mooncake 的"专家重排 + 同 batch 重做 forward"在物理上不成立：设备一重置，forward 状态必然丢失。
- FT 指南中的 `continue` 策略（幸存者不暂停继续服务）在 NPU 上结构性失效，只能走 `pause` 事务。
- sglang 请求层现状有一笔两种设备共同的欠账（FT 指南 NORMAL_FLOW_REVIEW F4）：scheduler 进程死亡后，in-flight 请求无 abort、无超时、无终态，客户端挂起到自身超时（`tokenizer_manager.py` 的 `update_process_active_ranks()` 只更新路由）。NPU 模型使这笔欠账变成必还项。

---

## 2. 兼容总原则：共用控制面 + 能力声明 + hook

控制面状态机、watchdog/lease、`pause → retry/scale_down → recover/rejoin` 协议、HTTP API **全部设备无关，两种设备共用，一行不删**。所有设备差异收敛到两个机制：

1. **设备容错能力声明**（数据）——描述该设备的恢复语义；
2. **设备容错操作 hook**（接口）——在固定钩点无条件调用，GPU 空实现，NPU 实现。

```text
┌──────────────────────────────────────────────────┐
│  FT 控制平面（设备无关, 共用）                       │
│  watchdog / lease / 状态机 / pause·retry·         │
│  scale_down·recover / admission / rejoin 协议     │
└───────────────┬──────────────────────────────────┘
                │ 能力声明 + hook 调用（无条件调用, 设备自决）
     ┌──────────┴───────────┐
     ▼                      ▼
 GPU/mooncake 实现           NPU/MC2 实现
 · hook 全部 no-op           · stop/restart_device
 · batch 原地保留             · reinit PG + 重算
 · membership 可 forward 观察 · 无 membership 通道
 · 图保持有效                 · 图按需重捕
     ▼                      ▼
 原地续推模型                 暂停-重置-重算模型
```

以下 7 个适配点覆盖全部架构差异。

---

## 3. 适配点

### 适配点 1：设备容错操作 hook 层

**差异**：故障后 NPU 必须 `stop_device`（清除挂起的 HCCL 状态）并在恢复时 `restart_device`；GPU 的 comm/mask 故障后依然有效，不需要任何设备操作。

**方案**：抽象 `FtDeviceOps` 接口，scheduler/rejoin 在三个固定钩点无条件调用，GPU 注册空实现，NPU 注册 torch_npu 实现（调用方式照抄 vllm-ascend `npu_worker_sentinel`）。控制面完全不感知设备。

```text
钩点                        GPU 实现      NPU 实现
─────────────────────────────────────────────────────────
pause 完成 ACK 后            noop          stop_device(本 rank device)
  (scheduler._complete_ft_pause 之后, 须先 synchronize)
resume 命令处理前            noop          restart_device + reinit PG
rejoin 进程 device init     noop          stop_device + restart_device
```

故障 rank 自己的设备不在 hook 覆盖内（进程已死，无人可调用）：由 rejoin 的 replacement 进程在自己的 init 钩点做 stop+restart 连做（同 vllm `clean_states`）；整节点消失则归外部 launcher——与 FT 指南既有边界一致，不新增机制。

### 适配点 2：in-flight 请求处理策略抽象（keep / discard / retract）

**差异**：这是两种模型的核心分歧。同一个故障：

- GPU membership 故障 → 请求原地保留（mooncake 同 batch 重做 forward）；
- GPU exception 故障 → 现状 discard + 503（`scheduler.py:1568-1628` `_ft_discard_inflight_window`）；
- NPU 任何故障 → batch 的 forward 状态无法存活，但请求**不该丢**，应回队重算。

**方案**：把"故障窗口内请求怎么处理"从硬编码变成策略枚举，由（设备能力 × 故障族）决定：

```text
                    keep 原地续推    discard 503    retract 回队重算
GPU membership 故障      ✓
GPU exception                       ✓
NPU 任何故障族                                    ✓
```

retract 分支**复用现成的 KV-OOM retract 原语**（`schedule_batch.py:2163` `retract_all` + `scheduler.py:3070` `_add_request_to_queue(is_retracted=True)`）：释放 KV、**保留 output_ids**、回 waiting queue，恢复后重 prefill 续推。客户端只感到停顿，无重复 token，等价于 vllm 的 preempt 语义。

实现上把 fault-window 清理函数参数化：窗口收集与状态清理（cur/last/running/result_queue/chunked 按 rid 去重、KV 尾部回退）两分支共用；discard 分支发 503，retract 分支回队不发终态。

```text
GPU:  [decode] ─故障─ [decode] ────→ [done]          原地, 无感
NPU:  [decode] ─故障─ 释放KV/留output ─回waiting─
      ─resume─ 重prefill ─ 继续decode → [done]       客户端只停顿
```

### 适配点 3：membership 事实源可插拔

**差异**：FT 状态机现在吃两个事实源：

```text
runtime_active = process_active ∧ mooncake_active
```

`mooncake_active` 来自 mooncake backend 在 survivor forward 中的 membership 观察上报。MC2 没有等价的 membership→控制面上报通道，这条源在 NPU 上不存在。

**方案**：把第二个事实源做成 **backend 注册的 observer**，而不是状态机的固定输入：

```text
runtime_active = process_active ∧ membership_active(若 backend 注册)

mooncake backend → 注册 forward 观察器（GPU 现状, 双源, 零改动）
MC2 backend      → 不注册（NPU 自动退化为单源, 只靠 watchdog/lease 静态检测）
```

`FaultToleranceState` 的派生逻辑一行不改。功能上 NPU 不损失：整节点消失本来就由 5s heartbeat lease 静态检测覆盖（不需要等 forward），只是少了"forward 驱动收敛"这条慢路径。

### 适配点 4：控制面集体通道可插拔 + resume 事务化

**差异**：两个相关问题。

1. NPU 的 `reinit_process_group` 是全员集体操作，resume 不能像 GPU 那样逐个 DP 翻标志完事。但**集体本身就是天然 barrier**（所有幸存 rank 不到齐就阻塞），不需要新设计 barrier 协议，只需 resume 命令覆盖全部幸存 rank——现有 `resume_targets = paused_dp_ranks ∩ runtime_active` 已满足。
2. 恢复后每次 forward 的 MLP sync 是跨 DP 集体。GPU 上它跑在 mooncake-cpu group（dead-rank-safe，天然容错）；NPU 上它跑在普通 gloo（`scheduler_components/dp_attn.py` 的 `prepare_mlp_sync_batch_raw`），含 dead rank 即 hang。这就是 MC2 方案 §4 的 C1 工程项，NPU 侧必须自建。

**方案**：把"故障后仍可集合通信的控制通道"抽象成组件，按 backend 注入：

```text
控制集体通道（MLP sync / 恢复期集体）
 ├─ GPU/mooncake: 现有 mooncake-cpu group（dead-rank-safe, 零改动）
 └─ NPU:          独立 FT gloo PG + stateless 重建 + 紧凑重编号
                  （照抄 vllm stateless_init_torch_distributed_process_group
                    机制, 图外重建不触发重捕）
```

resume 协议统一为四阶段事务：

```text
resume 命令（统一协议）
  ├─ 阶段1 device.restore()      GPU: noop    NPU: restart_device
  ├─ 阶段2 collective.rebuild()  GPU: noop    NPU: reinit PG（集体, 自然 barrier）
  ├─ 阶段3 batch.recover()       GPU: 保留    NPU: retract 回队（适配点 2）
  └─ 阶段4 _engine_paused=False  （两种设备共用）
```

GPU 走完前三步全是 no-op，行为与现状等价；resume ACK 语义变为"四阶段全部完成"。

### 适配点 5：恢复后的图处理 hook

**差异**：GPU 恢复不重建 comm，CUDA graph 固化的一切保持有效；NPU 重建 PG 后，graph 固化的 HCCL 句柄可能失效（MC2 方案"不重捕图"的前提是组不换，PG 重建打破了它）。

**方案**：恢复流程加一个"图处置"阶段，设备层声明恢复后图是否仍有效：

```text
device.graph_valid_after_recovery()
 ├─ true  → 跳过（GPU 恒真）
 └─ false → 在 pause 窗口内重建 graph pool, 再进 resume
```

NPU 侧先 PoC 验证句柄是否按 group name 延迟绑定（`CallHcomGetCommHandleByGroup` 的解析时机）：若有效则此 hook 对 NPU 也是 true，整个适配点关闭；若无效则重捕进 resume 事务。这是 NPU 侧最大的不确定项，但架构上它只是一个 hook，不影响共用代码结构。

### 适配点 6：故障信号入口统一（不新增第三故障族）

**差异**：GPU 的故障呈现是进程退出和 Python exception 两族。NPU 多出第三种呈现：**进程活着、设备 hang**（HCCL timeout、AI Core error）——watchdog 不报 DOWN，exception 路径也没人触发；若不处理，先撞上的是 Hard/SoftWatchdog（forward-progress 监控）的默认行为，与 FT pause 流程打架。

**方案**：不给控制面加第三故障族。在 NPU 的 forward 路径上加故障检查点（参照 vllm `evaluate_pause_condition → EngineLoopPausedError`），把设备 hang 转成受控异常，注入现有 exception 族入口：

```text
NPU 设备 hang → forward 检查点抛受控异常
  → _run_event_loop_fault_tolerance 捕获（现有, scheduler.py:1549-1566）
  → 请求策略 = retract（适配点 2, 不发 503）
  → FaultToleranceRankFaultOutput → 控制面 pause（现有）
```

控制面看到的仍然只有"exception 族"，设备差异消化在 model_runner 层。

### 适配点 7：故障 DP 请求的终态/重推（设备无关机制）

**差异**：这其实不是设备差异——进程死亡后 in-flight 请求无终态，GPU/NPU 同样存在（FT 指南 F4 欠账），只是 NPU 模型强制要求解决它。tokenizer_manager 的 `ReqState.obj` 保留了完整原始输入（`GenerateReqInput`），具备重放条件，但当前没有任何重入队逻辑。

**方案**：在 tokenizer_manager 层实现，完全设备无关，两种设备共用一套代码：

```text
新增 rid → DP 路由跟踪
process-DOWN 事件 → 扫描受影响 rid
 ├─ 未发出任何 token → 重推: 原 GenerateReqInput 发往健康 DP
 │    · 复用 rid + 重置 rid_to_state
 │    · 重推前/中检查客户端存活（is_disconnected）
 │    · 重试次数上限, 超限转 503（防连续故障死循环）
 └─ 已流式发出 token → 503 终态（重推会产生重复输出, 禁止）
```

终态唯一性：同一 rid 要么 503 要么最终结果，不得两者都发。NPU 默认启用；GPU 可作为选项启用（改善原生语义）或保持现状——机制只有一份。

---

## 4. 请求层契约（两种设备统一表述）

| 请求位置 | GPU/mooncake | NPU |
| --- | --- | --- |
| 健康 DP，membership 故障 | 原地续推，无感 | retract 回队重算，客户端只停顿、无重复 token |
| 健康 DP，exception 故障 | discard，503 | retract 回队重算 |
| 故障 DP，未发出 token | （现状：挂起；可选启用适配点 7） | 重推到健康 DP，拿到完整 response |
| 故障 DP，已流式发出 token | （现状：挂起） | 503 明确失败终态 |

"客户端 curl 没断就能拿到完整 response"只对**未发出 token 的重推请求**和**健康 DP 的 retract 请求**成立；已流式输出部分 token 的故障 DP 请求必须失败，否则客户端会收到重复/不连贯输出。

---

## 5. 总结

### 5.1 分层职责

| 层 | 共用（不动） | 分叉方式 |
| --- | --- | --- |
| 控制面 | 状态机 / watchdog / lease / pause·retry·scale_down·recover / admission / rejoin 协议 | 无 |
| 恢复协议 | pause→处置→resume 事务骨架 | 事务内各阶段 = 设备 hook，GPU no-op |
| 请求层 | fault-window 清理框架 / retract 原语 / 503·重推机制 | keep/discard/retract 策略枚举（设备×故障族） |
| 通信层 | MLP sync、resume 的调用位置 | 集体通道按 backend 注入（mooncake-cpu / 独立 FT gloo PG） |
| 设备层 | — | `FtDeviceOps` 实现 + 能力声明 |

一句话：**NPU 的全部特殊性收敛为"一份能力声明 + 三个设备 hook + 一个通道实现 + 一个策略枚举"，GPU 路径零删除零行为变更。**

### 5.2 NPU 侧需要自建的硬骨头

1. **独立容错 gloo PG**（适配点 4，即 MC2 方案 §4 的 C1 工程项）：stateless 重建 + 紧凑重编号，是 resume 事务和恢复后 MLP sync 的前置。
2. **图有效性 PoC**（适配点 5）：PG 重建后旧 NPU graph 句柄是否仍有效，直接决定恢复窗口是否需要重捕图，是最大的工作量变量。
3. **MC2 在新 PG 上 reinit 的最小闭环**：dispatch/combine 绑定新 compacted/rebuilt group 的 kernel/config 支持（见 compacted 成本文档 §6.1 的 `epWorldSize`、专家均分约束）。

retract 重算（适配点 2）与 503/重推（适配点 7）是设备无关机制，做一份两种设备都受益，同时也是 FT 指南 F4 产品契约欠账的偿还。

### 5.3 与 MC2 mask 路线的关系

本方案接受"stop/restart_device 是 NPU 硬件约束"的前提。MC2 方案中的 mask 容错（`x_active_mask` + 固定组 + 不重捕图）是另一条路线，两者前提互斥：

- 当前阶段按本方案实施（暂停-重置-重算模型）；
- 若未来 MC2 的 mask 能力成熟到进程 kill 类故障可以不重置设备，可演进为混合模式：**进程级故障走 mask 续推（GPU 语义），设备级故障走 stop/restart（本方案）**。本方案的 hook/能力声明架构不阻碍该演进。

### 5.4 明确不支持

- NPU 上的 `continue` 策略（幸存者不暂停继续服务）；
- pause/resume 进行中的二次故障（继承 FT 指南边界）；
- 已流式输出 token 的故障 DP 请求续推（会产生重复输出）。
