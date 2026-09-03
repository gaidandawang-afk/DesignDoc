# SGLang DP-only Fault Tolerance v0.1.0：统一架构与接口契约

> 规范级别：v0.1.0 目标控制面契约
>
> 当前落地差异：GPU 分支已实现本契约；NPU 静态缩容分支尚待 API/Manager 集成

## 1. 目标和非目标

本设计在一个 Data Parallel（DP）replica 发生 Python 异常、进程退出或节点租约失效后，将受影响流量与存活 replica 隔离，并由一个中心化控制面选择恢复、缩容或放弃受影响请求。

目标包括：

- 异常传播到同一 DP 的所有 scheduler rank；
- Manager 维护唯一、可审计的期望拓扑；
- 数据面拓扑变化与请求路由变化按 fail-stop 方式提交；
- GPU/Mooncake 和 NPU/MC2 可以使用不同的设备事务，但共享状态和外部 API；
- 已被缩容的 DP 在重新满足全部条件后可由控制面自动 rejoin；
- 每次操作可通过 `request_id` 区分，避免把历史拓扑误判为本次成功。

非目标包括：

- 单个 DP 内只缩掉部分 TP/attention-CP rank；
- 在本版本内自动选择恢复策略；
- 对失败请求提供透明、无损的 retract/replay；
- 用 SGLang admission 保护代替外部负载均衡器更新；
- 宣称 NPU 已具备与 GPU 相同的 retry、rejoin 或 Graph 恢复能力。

## 2. 术语和 rank 空间

设：

- `D`：DP replica 数；
- `R`：每个 DP 的 scheduler rank 数；
- `W = D × R`：全局 scheduler rank 数；
- `dp_rank`：稳定的原始 DP id；
- `global_rank`：Manager、watchdog 和进程存活 mask 使用的 scheduler rank id；
- `expected_dp_mask[D]`：控制面期望仍参与服务的 DP；
- `process_alive_global_rank_mask[W]`：进程/节点租约观察结果；
- `native_active_dp_mask[D]`：Mooncake 或 MC2 数据面报告的可服务状态；
- `route_dp_mask[D]`：最终 admission/routing 可用集合。

whole-DP 约束要求 DP `d` 对应的全部 `R` 个 global rank 被原子地视为一个故障域。不能因为其中一个进程仍活着就继续向该 DP 路由。

稳态下一个 DP 可被路由的必要条件是：

```text
serving_ready[d]
  = expected_dp_mask[d]
  AND all_processes_of_dp_d_are_alive
  AND native_active_dp_mask[d]
```

任何一项为 false，该 DP 都不能接收新请求。当前 `continue` 路径按这个公式更新 `_route_dp_mask`；`pause` incident 期间 `_route_dp_mask` 可能暂时保留旧值，但 `cluster_paused` 或 `ft_operation_in_progress` 会全局拒绝 admission。因而 `_route_dp_mask` 本身不是唯一服务事实，不能脱离 pause/operation guard 单独解释。

continue 下的 Mooncake membership loss 不提交控制面缩容，`expected_dp_mask` 保持 true。runtime-active 或 process-alive 变为 false 已足以关闭 route；两者恢复后 route 自动恢复。因此 expected 表示控制面期望拓扑，不表示当前 serving 集合。

## 3. 控制面组件和唯一事实源

### 3.1 Manager

Manager 是 FT 操作的唯一提交者，负责：

- 汇总 scheduler 异常和节点 heartbeat lease；
- 维护 expected、process-alive、native-active、pending-recovery mask；
- 将故障扩展为 whole-DP 范围；
- 串行执行 `retry` 或 `scale_down`；
- 发布路由 mask；
- 记录最近一次 FT `request_id` 和错误；
- 在进程及数据面恢复后自动 rejoin。

Scheduler 或 HTTP handler 不得绕过 Manager 独立提交拓扑。

### 3.2 Scheduler

Scheduler 负责本地请求与模型执行状态，并通过内部控制消息执行：

- 进入/退出 FT pause；
- 丢弃异常时受影响的请求；
- 调用 ModelRunner/backend 的恢复或 scale-down 操作；
- 上报进程和 native backend 状态。

### 3.3 设备数据面

设备数据面执行后端专属事务：

- GPU/Mooncake：更新 active rank、恢复连接、处理 EPLB/DeepEP 状态；
- NPU/MC2：stop/restart device，创建 compact survivor Gloo/HCCL 组，更新 EPLB 映射和 MC2 `elastic_info`。

控制面只在数据面事务成功后提交新的 expected/route 状态。若执行结果不确定，保持隔离并 fail-stop。

## 4. 对外状态机

### 4.1 公开状态

公开状态固定为三种：

| 状态 | 含义 | 可接收新请求 |
|---|---|---|
| `healthy` | 当前 classifier 中：仍 expected、进程完整且不在 `unhealthy_dp_ranks` | 取决于 route/native/pause |
| `unhealthy` | 仍 expected，且 pause 策略记录到未处理 scheduler incident | 否 |
| `dead` | 已不在 expected 集合，或 DP 的进程不完整 | 否 |

`disabled` 是旧文档状态，v0.1.0 不再使用。被静态缩容的 DP 显示为 `dead`。

必须注意：当前 status classifier 不直接读取 `native_active_dp_mask`，而 admission、continue route 和自动 rejoin 会读取它。因此 status 中的 `healthy` 不是“此刻一定可路由”的同义词；操作确认仍需同时检查 route 行为和目标 topology。若后续产品要求 status 精确表达 native-not-ready，需要单独调整 classifier，不能只改文档。

### 4.2 内部状态

公开三态不足以驱动恢复。当前 Manager 内部至少维护：

```text
expected_dp_mask
native_active_dp_mask
process_alive_global_rank_mask
pending_recovery_global_ranks
unhealthy_dp_ranks
ft_operation_in_progress
cluster_paused
```

这些字段的职责不同：

- expected 是控制面已提交意图；
- process-alive 是 watchdog/进程事实；
- native-active 是设备后端事实；
- pending-recovery 防止进程刚出现但数据面尚未恢复时提前放流；
- unhealthy 表示仍需处置的 incident；
- operation guard 保证单事务；
- cluster-paused 描述当前请求策略。

不得用一个复合 `disabled` 标志代替这些独立事实。

### 4.3 典型转换

```text
healthy
  ├─ pause 下的 Python/collective incident ─────> unhealthy
  │                                                ├─ retry 成功 ──> healthy
  │                                                ├─ scale_down ─> dead
  │                                                └─ 操作失败 ───> unhealthy / fail-stop
  ├─ continue 下的 Mooncake membership loss ────> expected 不变
  │                                                └─ runtime/process 观察更新 route/status
  └─ process/lease lost ───────────────────────> dead
                                                    └─ process + native 恢复
                                                       且无 pending ─> healthy
```

`dead -> healthy` 是自动 rejoin，不是显式 `recover` API。

## 5. HTTP API

### 5.1 查询状态

```http
GET /fault_tolerance/status
```

目标响应：

```json
{
  "schema_version": 1,
  "total_engines": 4,
  "engines": [
    {
      "id": 0,
      "status": "healthy",
      "last_ft_request_id": "ft-20260826-001"
    },
    {
      "id": 3,
      "status": "dead",
      "last_ft_request_id": "ft-20260826-001"
    }
  ]
}
```

字段约束：

- `id` 使用稳定的原始 DP id，不使用 compact group rank；
- 成功和失败都可显示最近处理的 `last_ft_request_id`；
- `ft_error` 只在对应操作失败时出现；
- status endpoint 在 DP 不可推理时仍应可访问，以便诊断；
- 客户端必须容忍 `message`/`detail` 作为不同 HTTP 层错误 envelope，但不能放松状态和 request id 断言。

### 5.2 提交操作

```http
POST /fault_tolerance/apply
Content-Type: application/json
```

`retry`：

```json
{
  "instruction": "retry",
  "params": {},
  "request_id": "ft-20260826-002"
}
```

`scale_down`：

```json
{
  "instruction": "scale_down",
  "params": {
    "removed_dp_ranks": [3]
  },
  "request_id": "ft-20260826-003"
}
```

合法 envelope 被异步接受时返回 HTTP 202：

```json
{
  "message": "Request accepted; poll /fault_tolerance/status for updates.",
  "request_id": "ft-20260826-003"
}
```

HTTP 202 只表示进入控制面队列，不表示数据面事务已经成功。

### 5.3 完成判定

客户端轮询必须同时满足：

1. 目标 engine 的 `last_ft_request_id` 等于本次提交值；
2. `ft_error` 不存在；
3. engine 状态和目标 topology 满足本次操作的终态。

例如旧操作已经把 rank 3 缩掉，本次无效请求不能因为 rank 3 仍为 `dead` 就被判成功。

### 5.4 同步错误和异步错误

| 层级 | HTTP/状态 | 示例 |
|---|---|---|
| envelope/类型错误 | 400 | body 非对象、未知 instruction、`params` 非对象、request id 非字符串 |
| 另一个 FT 操作执行中 | 409 | operation guard 冲突 |
| FT 功能未启用 | 503 | 当前服务没有 FT 控制面 |
| 合法请求已接受 | 202 | 后续通过 status 判定 |
| 语义前置条件失败 | 202 后 `ft_error` | 无 incident、移除全部 DP、目标已不 expected |

语义错误异步化的原因是保证写路径只有一个提交者。客户端不可只检查 POST 状态码。

当前实现允许省略 `request_id`，此时使用空字符串；调用方应主动提供唯一非空值。`scale_down` 的 instruction-specific 参数也尚未在返回 202 前完整校验，详见 [VALIDATION.md](VALIDATION.md) 的待决代码问题。

### 5.5 删除的旧接口

以下均不是 v0.1.0 目标契约：

- `recover` instruction；
- `params.ranks`；
- 客户端传入 FT timeout；
- 同步等待数据面完成后返回 HTTP 200；
- `{"ranks":[{"rank":0,"state":"..."}]}` status schema。

NPU 分支当前仍包含这些旧接口，属于待移植差异，不是本规范的兼容别名。

## 6. 故障类型与处置

### 6.1 Mooncake membership loss

pause 和 continue 在推理层的处理方式不同：

- `pause`：`forward_raw` 后检测到 active-rank 减少时主动抛出异常，Scheduler 进入 pause，Manager 记录 `unhealthy`，等待显式 `retry` 或 `scale_down`；
- `continue`：该路径不抛异常。Mooncake Elastic EP 自动完成 active-rank 4→3，EPLB 随后收敛，ModelRunner 再次执行 `forward_raw`。Manager 不创建或下发 FT 事务，只根据 runtime/process 上报更新 route 和 status，`expected_dp_mask` 保持不变。

NPU v0.1.0 只声明 pause 后执行静态 scale-down 的代码路径。

### 6.2 进程退出或节点 lease 失效

Manager 将该节点上广告的 global ranks 标记为不存活，再扩展为 whole-DP 影响范围。其 DP 对外状态为 `dead`，路由立即关闭。

watchdog 观察到的是逻辑节点 heartbeat lease，而不是电源状态本身。真实整机下电、网络分区和同机多逻辑节点的租约超时应在验证报告中分开描述。

### 6.3 `retry`（GPU）

`retry` 仅适用于 pause 下仍 expected 的 unhealthy DP，且所有 expected 进程必须存活。continue 的 Mooncake 自动收缩和 forward 重试不经过该控制面事务。典型步骤：

1. operation guard 获取事务；
2. 验证存在 incident，且 expected 进程完整；
3. 保持相关路由关闭；
4. 通知 scheduler/backend 清理失败状态并重试；
5. 等待 native active 和请求处理状态收敛；
6. 重新发布 route；
7. 写入本次 `last_ft_request_id`。

NPU MC2 分支的 retry 路径没有执行 device stop/restart、survivor group 重建或 elastic-info 更新，不能视为支持。

### 6.4 `scale_down`

通用前置条件：

- 已有待处理 incident；
- `removed_dp_ranks` 非空、无重复、均仍 expected；
- 缩容后至少保留一个 DP；
- 目标扩展到 whole-DP global ranks；
- 不对 rank0 注入故障是当前稳定测试约束，不等价于 API 永久禁止移除 DP0。

通用事务顺序：

1. 验证请求并冻结相关路由；
2. 终止/隔离目标 DP 的全部 scheduler rank；
3. 只向存活 rank 下发新的 active topology；
4. 在设备后端执行不可分割的重建/更新事务；
5. 等待存活 rank 报告 native active；
6. 提交 `expected_dp_mask=false`；
7. 发布新的 route，并完成 request id。

被移除 rank 不接收 scale command，其 EngineCore 保持 `DEAD`。这和 vLLM 的 survivor-only 设计一致，但被移除 endpoint 的推理 HTTP 状态码由 SGLang admission 契约定义。

## 7. 后端事务

### 7.1 GPU/Mooncake

GPU 使用稳定原始 rank 命名空间和 active mask。Mooncake 负责：

- 屏蔽 inactive peers；
- 本地 NVLink/IPC fast path；
- 延迟远端连接和 rejoin；
- EPLB/DeepEP 状态随 active ranks 收敛；
- 在 CUDA Graph 捕获前准备 local-only fast path，捕获期间跳过 inactive peer。

详见 [GPU_MOONCAKE.md](GPU_MOONCAKE.md)。

### 7.2 NPU/MC2

NPU 当前分支没有复刻 Mooncake 的固定通信组实现，而是：

- 为存活 DP 创建 compact Gloo MLP-sync group；
- 为存活 DP 创建 compact HCCL EPLB group；
- 保留 original DP id 作为控制面 id；
- 通过映射将 original rank 转为 compact group rank；
- 通过固定地址、原地更新的 `elastic_info` 向 MC2 dispatch/combine 传递 effective EP topology。

该方案只重建新引入的专用通信组，不会替换整个 SGLang 全局 `_TP` group。详见 [NPU_MC2_STATIC_SCALE_DOWN.md](NPU_MC2_STATIC_SCALE_DOWN.md) 和 [NPU_MC2_COMPACTED_TOPOLOGY.md](NPU_MC2_COMPACTED_TOPOLOGY.md)。

## 8. 路由恢复和自动 rejoin

continue 的普通 4→3→4 不改变 expected topology。runtime/process 观察重新满足 serving-ready 条件后，Manager 直接恢复 route，不经过 `_auto_recover_ready_dps()`。

只有已经由控制面 scale-down、`expected_dp_mask == false` 的 DP 才需要自动 rejoin。其开放条件是：

```text
expected == false
process alive for all global ranks
native backend active
no pending recovery rank
```

实现完成内部恢复后，Manager 将 expected 重新置为 true 并重新发布路由，不要求外部调用 `recover`。

必须区分：

- 进程重新注册：只证明 alive；
- backend 连接恢复：证明 native active；
- admission 重开：证明 route 已提交；
- 推理精度和持续请求通过：证明端到端恢复。

NPU 分支尚未打通上述完整链路，尤其 compact group 代际、device restart 与外部进程 rejoin 的交互没有硬件验证。

## 9. 路由和负载均衡

### 9.1 SGLang admission

普通调度只能从 `route_dp_mask` 选择 DP。显式 `routed_dp_rank` 也必须经过相同检查；目标 inactive 时返回：

```text
HTTP 503
routed_dp_rank=<N> is not active
```

不得返回 HTTP 200 + abort finish reason 作为目标契约。该行为曾在待测实现中被观察到，但不是 vLLM 上游 PR 已定义的接口。

### 9.2 外部 LB

外部 LB 必须在 scale-down/rejoin 后更新 endpoint 集合。SGLang 503 用于防止状态传播延迟或显式错误路由造成静默误服务，不意味着 LB 可以持续向 dead endpoint 发送请求。

## 10. 安全边界和 fail-stop

下列情况必须保持隔离，不可“尽力而为”继续：

- survivor backend 重建部分成功、部分失败；
- process alive 与 native active 不一致；
- collective 超时且无法证明所有 survivor 到达同一代际；
- compact rank 映射与 original rank mask 尺寸不一致；
- request id 已完成但目标 topology 不一致；
- Graph 捕获引用了 inactive peer 或未准备的通信资源。

系统可暴露诊断 status，但不得在状态不确定时重新开放推理路由。

## 11. 配置门槛

当前 GPU 实现的关键门槛包括：

- DP 数大于 1，启用 DP attention；
- 启用 EPLB；
- `elastic_ep_backend=mooncake`；
- pipeline parallel size 为 1；
- 不与运行时 EP scale、PD disaggregation、Ray 多节点编排等未集成模式组合；
- whole-DP 拓扑可整除并能从 global rank 精确映射回 DP。

NPU 分支将 `device=npu`、`elastic_ep_backend=mc2` 加入旧 support gate，但尚未合并当前 GPU 分支的完整 gate。v0.1.0 目标规格允许 `attn_tp>1`：whole-DP mask 已按 global scheduler rank 展开，survivor compact group 也可以直接使用该空间。核对提交仍需把通信对象从 `dp_rank/dp_size` 改为 `tp_rank/tp_size`，并把 MLP-sync 结果写入展平的 global-rank 槽位；预计少于 10 行产品代码。完成该改动和 NPU E2E 前，能力等级仍是“设计支持、当前提交未验证”。

## 12. 规范性约束摘要

1. Manager 是 FT 操作唯一提交者。
2. 容错和缩容单位固定为 whole DP。
3. 对外只有 `healthy`、`unhealthy`、`dead`。
4. 写操作只有 `retry`、`scale_down`。
5. 合法写请求返回 202，终态由 request id + topology 确认。
6. 同一时间最多一个 FT 操作。
7. `scale_down` 只通知 survivor，不通知 removed rank。
8. inactive DP 的显式推理路由返回 503。
9. route 必须同时受 expected、alive 和 native-active 约束。
10. rejoin 是条件满足后的自动提交，不暴露 `recover`。
11. 后端事务可以设备特化，但失败必须 fail-stop。
12. 测试结论必须绑定精确源码/依赖提交和 artifact。
