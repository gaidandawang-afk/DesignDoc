# GPU + Mooncake DP-only Fault Tolerance 实现说明

> SGLang：`codex/ft-vllm-api-refactor` @ `90380d6e6ce7bdefc3d209f4fe284623d40720fb`
>
> Mooncake：`codex/mooncake-nohca-ft` @ `d727290c86e4c821a6fc4c22848ae5c0f269f4f5`
>
> 核对日期：2026-08-26

## 1. 实现范围

GPU 路线是 v0.1.0 中完成度最高的实现，覆盖：

- scheduler exception 与进程/逻辑节点丢失；
- `pause`、`continue` 两种 incident 策略；
- `retry` 与静态 `scale_down`；
- whole-DP shutdown；
- 存活 DP 持续服务；
- inactive DP 显式路由拦截；
- process + Mooncake native 双条件自动 rejoin；
- EPLB/DeepEP 弹性状态恢复；
- CUDA Graph 下延迟连接和 inactive peer 跳过；
- 操作级 `request_id` 状态确认。

这不是 upstream SGLang/Mooncake 已发布能力的泛化声明。两个待测分支及其提交必须成对记录。

## 2. 代码分层

### 2.1 SGLang 控制面

主要实现位于：

- `python/sglang/srt/fault_tolerance/manager.py`：事务、状态、watchdog、route、rejoin；
- `python/sglang/srt/fault_tolerance/controller.py`：状态计算和 whole-DP mask；
- `python/sglang/srt/managers/data_parallel_controller.py`：DP controller shutdown 和控制消息；
- `python/sglang/srt/managers/tokenizer_manager.py`：Manager 与请求入口衔接；
- `python/sglang/srt/entrypoints/http_server.py`：status/apply HTTP endpoint；
- scheduler、ModelRunner、EPLB/DeepEP 相关模块：执行后端命令并回报 active 状态。

控制面不依赖 HTTP handler 同步等待。`submit()` 设置全局 `ft_operation_in_progress`，创建后台任务并立即返回 202；内部命令使用独立 request id 收集 scheduler ack 和 route update ack。

### 2.2 Mooncake 数据面

Mooncake 分支扩展弹性 EP 连接与恢复，重点不是重建 Python 进程，而是让同一 rank 命名空间中的 survivor 能安全跳过 inactive peers，并让重启 rank 恢复本地/远端连接。

当前依赖中两个关键修复是：

- `1ca33d50`：在 `connect_on_init=false` 时，Graph 捕获前为 replacement Buffer 安装本地 IPC handle 和 local-only active mask；
- `d727290c`：dispatch/combine Graph 捕获期间根据 active ranks 跳过 inactive peer，避免捕获到尚未建立的连接。

## 3. Mooncake 自动收缩与控制面观察

### 3.1 continue

continue 的故障处理与 Mooncake Elastic EP 保持一致。`forward_raw` 返回后发现 active-rank 从 4 减少到 3 时不抛异常；Mooncake 更新 active ranks，EPLB 完成收敛，ModelRunner 再次执行 `forward_raw`。

这条路径没有控制面 FT 请求，也不产生 `unhealthy` incident。`expected_dp_mask` 保持 true；Manager 只消费 runtime/process 上报，并用 observed availability 更新 status 和 route。

### 3.2 pause

pause 下，`forward_raw` 后检测到 membership loss 会主动抛异常。Manager 记录 `unhealthy`，集群 admission 返回 `fault_tolerance_paused`，直到显式 `retry` 或 `scale_down` 完成。

### 3.3 进程与逻辑节点故障

watchdog 维护 `node_id -> (last_seen, advertised_global_ranks)` 的 lease。当前参数为：

```text
sweep interval = 1 second
lease timeout  = 5 seconds
```

lease 超时后，该节点广告的全部 global ranks 被标为 down，再按 whole-DP 映射关闭相关路由。代码入口为 `FaultToleranceManager.observe_watchdog_heartbeat()`。

现有 E2E 日志已经观察到：

```text
FT watchdog lease expired: nodes=[3] global_ranks=[3]
```

但这组 E2E 在同一物理服务器上启动多个逻辑节点/进程，证明的是租约和 whole-node 聚合逻辑，不是拔电、主机 kernel panic 或真实跨机网络隔离。单元测试另覆盖一个逻辑节点广告 `[2, 3]` 后两者同时 down。

## 4. `retry` 事务

`retry` 用于 pause 下进程仍存活、DP 仍 expected 的异常；continue 的自动 forward 重试不经过该事务。前置条件：

- `unhealthy_dp_ranks` 非空；
- 每个 expected DP 的全部 global ranks 都存活；
- 没有另一个 FT 操作执行中。

执行顺序：

1. 向全部 expected DP 发送 `retry`；
2. 等待所有目标 ack；
3. 发布 expected route mask 并等待确认；
4. 清除 unhealthy incident；
5. status 记录本次 `last_ft_request_id`。

任何底层命令/ack 异常走 `_fatal_task_wrapper()` 和 fail-stop，不会降级成“部分 retry 成功”。此时 operation guard 可能保持占用，目的是阻止未知拓扑继续接受新控制事务。

## 5. `scale_down` 事务

### 5.1 控制面顺序

当前实现按以下顺序执行：

1. 验证已有 incident；
2. 将 `removed_dp_ranks` 转为集合；
3. 验证目标仍 expected，且不能移除全部 expected DP；
4. 请求 DataParallelController 关闭目标 DP 的全部进程；
5. 只向 survivor DP 发送 `scale_down`，携带 whole-DP 展开的 global active mask；
6. 等待 survivor 数据面收敛；
7. 发布 candidate route mask；
8. 提交 expected mask，并记录 request id。

removed DP 不会收到 scale-down command。它的 EngineCore 保持 `DEAD`，只有 status/诊断入口需要继续可访问。

### 5.2 rank0 测试边界

API 当前没有永久硬编码“DP0 不可缩容”。但稳定用例明确避免向 rank0 注入故障或将其作为 removed target，因为 Mooncake coordinator/rendezvous 的所有权仍与 rank0 耦合。

文档和用例必须区分：

- API 输入域：没有在 envelope 层禁止 0；
- 当前可靠验证域：只对非 rank0 注入/缩容；
- 后端工程风险：rank0 coordinator 生命周期尚未解除耦合。

在没有独立验证前，不应以“代码接受 0”宣称 rank0 缩容受支持。

## 6. admission 和外部 LB

显式请求携带 `routed_dp_rank` 时，Tokenizer/Manager 在进入推理前检查 route mask。inactive target 返回：

```text
HTTP 503
routed_dp_rank=<N> is not active
```

该行为由 SGLang `7375c482ba` 固化并已有场景验证。

[vLLM PR #46370](https://github.com/vllm-project/vllm/pull/46370) 的契约只要求 scale-down 命令发给 survivor，被移除 EngineCore 保持 DEAD，并验证 survivor 恢复后 `/v1/completions` 返回 200；它没有规定对 removed endpoint 发推理请求必须返回 HTTP 200 或某种 finish reason。因此：

- SGLang 503 是本实现的 admission 契约；
- 外部 LB 必须移除 dead endpoint；
- 不能把旧观察到的 `HTTP 200 + type=abort` 写成上游 vLLM 保证。

## 7. 自动 rejoin

continue 下发生普通 4→3→4 时，expected 始终为 true。`_update_route_after_observation()` 仅在 process alive、native active 且 pending recovery 已清空后恢复 route；任一观察尚未收敛时，status 保持 `dead`，route 保持关闭。

被控制面 scale-down、expected 已经为 false 的 DP 重新出现后，Manager 不立即放流。它分别观察：

1. 该 DP 全部 global process alive；
2. Mooncake native active；
3. `pending_recovery_global_ranks` 已清空；
4. 当前没有 FT 操作执行中。

全部满足后，`_restore_ready_dps_to_expected_topology()` 将 expected 重新置 true，`_update_route_after_observation()` 发布新 route。该自动路径用于恢复已提交移除的 DP，没有外部 `recover` 指令。

rejoin 验证不能只检查 status 变绿，还应覆盖：

- 被移除时显式路由返回 503；
- 重启进程注册；
- native mask 恢复；
- route 重新包含该 DP；
- 显式路由推理成功；
- MoE 精度符合基线；
- 连续请求没有因旧连接或 stale Graph 失败。

## 8. Mooncake no-HCA 恢复

### 8.1 稳定原始 rank + active mask

GPU 路线保持原始 rank namespace，不为每次 shrink 重编号。survivor 的 kernel/transport 根据 active mask 跳过失效 peer；重新加入的 rank 在恢复必要连接后重新变 active。

这样控制面 id、外部 `routed_dp_rank`、EPLB 元数据和诊断日志使用同一稳定 id，避免把 compact group rank 暴露到 API。

### 8.2 本地 fast path 和延迟远端连接

no-HCA/节点内路径依赖 NVLink/IPC。本地通信资源必须在可能发生 Graph 捕获前准备；远端连接可在恢复阶段延迟建立。

若 replacement Buffer 使用 `connect_on_init=false` 却没有本地 fast path，Graph 捕获会接触未准备 peer。旧依赖组合曾出现：

```text
cudaErrorStreamCaptureUnjoined
```

该问题不是 FT API 重构导致，而是 SGLang 与 Mooncake 分支配对不完整。

### 8.3 CUDA Graph 的两个必要修复

`1ca33d50` 先准备 local-only fast path，使捕获前已有本地 IPC 资源；`d727290c` 再让 kernel 在捕获期间跳过 inactive peers。二者共同满足：

- Graph 捕获不等待 inactive rank；
- Graph 中使用的本地通信地址有效；
- rejoin 后 backend 可以恢复远端连接；
- replay 不引用 stale/inactive peer。

单独升级 SGLang API 分支无法弥补缺失的 Mooncake 修复。

## 9. 支持 gate

当前 GPU 分支只在以下组合中启用 FT：

- `dp_size > 1`；
- DP attention 开启；
- EPLB 开启；
- `elastic_ep_backend=mooncake`；
- pipeline parallel size 为 1；
- 不使用运行时 EP scale/join 模式；
- 不使用 PD disaggregation；
- 不使用 Ray 数据并行控制器；
- tokenizer 数及 whole-DP 拓扑满足当前实现约束；
- 当前分支显式拒绝 NPU，因为 NPU port 尚未合并。

gate 是启动期 fail-fast，不是未来支持路线的否定。任何扩大矩阵都必须新增对应契约和 artifact。

## 10. 验证摘要

server_tool 的 16 个场景覆盖：

| 类别 | 覆盖内容 |
|---|---|
| 基线 | FT 关闭、原生请求、status-only |
| retry | 单异常 retry、先缩容后 survivor 异常 retry |
| scale-down | pause、异常触发、double/continuous shrink |
| topology | whole DP shutdown、A×C>1、4→3→2→1 |
| API | envelope、busy、前置条件、inactive route 503 |
| rejoin | continue、pause、CUDA Graph |
| 请求语义 | in-flight kill、Mooncake forward 重试、unattended pause timeout |

精确断言数、提交配对和 artifact 见 [VALIDATION.md](VALIDATION.md)。

## 11. 已知边界

- 当前没有真实物理节点下电/网络分区的独立 artifact；
- rank0 故障/缩容不在稳定验证域；
- pipeline parallel、PD、Ray、非 Mooncake EP backend 不支持；
- NPU 不能直接使用本 GPU 分支结论；
- `request_id` 当前只校验类型，省略时会使用空字符串；调用方应始终提供全局唯一非空值；
- instruction-specific params 的 envelope 校验仍不完整，畸形 `removed_dp_ranks` 可能进入后台 fail-stop，见 [VALIDATION.md](VALIDATION.md) 的待决问题；
- 16 个场景的证据跨两个相邻 SGLang 提交：15 个在 `7375c482ba`，CUDA Graph/status 增量在 `8a8860da82`；尚未在单一 `8a8860da82` 上重跑全量 16 场景。

## 12. 依赖配对规则

| SGLang | Mooncake | 结论 |
|---|---|---|
| `7375c482ba` | `1569df4d` | CUDA Graph rejoin 失败；缺 deferred local fast path/inactive peer 处理 |
| `8a8860da82` | `d727290c` | CUDA Graph rejoin 78/78 断言通过 |

每次报告必须记录两边完整提交。只写“最新 SGLang + 最新 Mooncake”不可复现，也不能作为证据。
