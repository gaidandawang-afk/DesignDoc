# SGLang Fault-Tolerance 设计文档 v5（架构设计）

> **版本**：v5.1.4  
> **日期**：2026-06-18  
> **目标读者**：后端开发工程师、测试工程师、SRE、技术负责人  
> **配套文档**：`sglang_fault_tolerance_design_v5_noft_impl_zh.md`  
> **参考基线**：v3/v4 设计文档、`D:\Codex\repos\sglang` 的 noFT baseline 分支 `codex/mooncake-noft-baseline-e63cb37b`

---

## 1. 范围与结论

### 1.1 文档目标

本文档冻结 SGLang FT v5 的外部语义、故障模式、行为矩阵、状态机和控制面职责。v5 按真实故障处理策略重写，不沿用 v4 中“recoverable / non-recoverable + continue 自动隔离”的抽象。

核心结论：

1. 故障输入只有 `exception`、`kill`，另单独处理 Mooncake 底层屏蔽 rank 的 `inactive_rank`。
2. 对外接口只有 `status` 和 `apply`；`retry`、`scale_down` 是 `apply` 的 instruction 参数。
3. 行为模式只有 `noFT`、`FT_pause`、`FT_continue`。
4. 对外只暴露 DP rank。v5 初版只支持实际模型计算拓扑中的 effective TP=1；这里的 TP 不是 CLI 原始 `--tp-size` 字面值。MoE/Mooncake/DP-attention 场景下，启动参数 `--tp-size>1` 可能表示全局 world、EP 或 DP-attention 组合维度，不等价于模型张量切分 TP，不能据此直接拒绝启动。
5. 实现不维护复杂实例状态机，只维护 rank 状态和一个内部 `ft_operation_in_progress` 标记。
6. `retry` 和 `scale_down` 共用恢复动作：先执行容错动作并收集 ack，再提交 DPC routing 和对外 FT status。

### 1.2 v4 是否满足新场景

结论：**不满足，需要 v5 重写状态机和实现路径。**

| v4 设计点 | 与真实场景的冲突 | v5 修正 |
|-----------|------------------|---------|
| `FT_continue + recoverable_fault` 设计为 fault rank paused、healthy ranks 继续服务 | 新语义要求 `FT_continue + exception` 上报异常但控制面不处理，不改变 status | scheduler 可以上报 exception event；控制面只记录日志/metrics，不 pause、不改 rank 状态 |
| `FT_continue + kill` 自动隔离 dead rank | 新语义要求表现和 noFT 一样，同时 FT status 标记 dead | 不暂停、不等待人工恢复；自动更新 routing / active mask 以避开 dead rank |
| `retry` 拒绝 dead rank | 新语义中 `FT_pause + kill` 后允许 retry，效果和 scale_down 指定已有 dead rank 一样 | `retry` 不理解 dead 来源，只按当前 `dead` mask 恢复 `paused` ranks |
| `scale_down` 使用 prepare/reinit/health_check/resume 序列 | 新语义中 `scale_down` 是两阶段恢复动作 | `scale_down = compute target mask + apply/resume + optional DPC-local shutdown + commit status` |
| 控制面维护 `RUNNING/PAUSED/DEGRADED/RECOVERING/FAILED` | 实际只需要判断是否正在处理故障、是否存在 paused ranks | 实现只保留 `ft_operation_in_progress`；状态机仅作文档辅助理解 |
| FT 主动清理/abort 请求 | exception 与 kill 分开处理 | exception 丢弃当前 scheduler batch、释放 KV 并向客户端返回 503；kill 不主动清理 surviving ranks 的 in-flight 请求，行为对齐原生 Mooncake |
| 未把 Mooncake inactive rank 作为独立输入 | 新语义要求新增故障 rank 时，`FT_continue` 更新 status，`FT_pause` 主动 pause | TokenizerManager 截取 `ActiveRanksOutput` 的 true->false 作为 `inactive_rank` 输入 |

### 1.3 对外接口与动作

| 类型 | 内容 | 说明 |
|------|------|------|
| HTTP API | `GET /fault_tolerance/status` | 查询 DP rank 的 `healthy|paused|dead` |
| HTTP API | `POST /fault_tolerance/apply` | 执行 `retry` 或 `scale_down` |
| CLI 参数 | `--enable-fault-tolerance` | FT 总开关 |
| CLI 参数 | `--fault-tolerance-on-error-strategy pause|continue` | FT 开启时的故障策略 |

`retry` 与 `scale_down` 的差异：

```text
retry:
  不接受参数
  不新增 dead rank
  使用当前 status 中已有 dead ranks 生成 active_mask
  resume 所有 paused ranks

scale_down:
  接受 ranks 参数
  先将参数指定 DP ranks 标为 dead
  再使用更新后的 status 生成 active_mask
  resume 所有 paused ranks
```

因此 `FT_pause + kill` 后允许 `retry`：

```text
kill rank1:
  rank1 = dead
  rank0/2/3 = paused

apply retry:
  active_mask = [true, false, true, true]
  resume rank0/2/3
  result: rank0/2/3 healthy, rank1 dead
```

这与 `apply scale_down ranks=[1]` 的结果相同。更精确地说，`retry` 不接受用户指定的新拓扑变更，但会按控制面已经观测到的 `dead` 状态恢复服务。

### 1.4 非目标

- 不实现对外 `pause` API。
- 不实现 rank replacement 或新进程热加入。
- 不实现 NCCL compact-rank rebuild。
- 不承诺非 Mooncake 后端的 `scale_down` 恢复。
- 不支持实际模型计算拓扑 effective `TP>1` / `PP>1` / 跨节点 FT 初版语义；FT enabled 时启动拒绝。禁止用 raw CLI `--tp-size>1` 直接推断 effective TP>1，尤其不能误拒 MoE/Mooncake 合法启动形态。
- DP attention 是既有默认启动形态，v5 初版不因 DP attention 直接拒绝；FT 代码只依赖 DP rank 级 routing/status，不主动理解 DP attention 内部细节。
- 不承诺请求级重放或 exactly-once。exception 丢弃当前 batch；kill 的 in-flight 请求对齐原生 Mooncake。
- 暂假设单机部署。

### 1.5 已确认的设计点

| 议题 | v5 结论 |
|------|---------|
| 是否需要 `dead_isolation_applied` | 不需要，删除。控制面只看 `healthy|paused|dead`。 |
| 是否需要实例状态 | 实现不需要复杂实例状态；只需要内部 `ft_operation_in_progress`。文档中的状态机只辅助理解。 |
| `FT_continue + exception` | 上报异常事件；控制面不处理、不改变 status；scheduler 预留 no-op hook 后继续 event loop。 |
| `FT_pause + kill + retry` | 允许。retry 使用当前 dead mask 恢复 paused ranks。 |
| status rank 粒度 | 只暴露 DP rank；v5 初版仅支持实际模型计算拓扑 effective TP=1，不能按 raw `--tp-size` 判断。 |
| 带 FT 代码但不开启特性 | 必须与无 FT 代码的 noFT kill 原生隔离行为一致，这是验收项。 |

---

## 2. 术语与状态机

### 2.1 术语表

| 术语 | 类型 | 定义 | 辨析 |
|------|------|------|------|
| FT rank | 名词 | FT 控制面的状态单位。v5 初版等同 DP rank / scheduler 组 | 初版 effective `TP=1`，一个 scheduler 进程等于一路 DP |
| `healthy` | rank 状态 | rank 可达、未被隔离、可参与正常 event loop 和普通推理 | vs `paused` / `dead` |
| `paused` | rank 状态 | rank 存活且可控，但停止正常 event loop，等待 `resume` | 只接收 FT safe control |
| `dead` | rank 状态 | rank 已退出、不可达、被 Mooncake 屏蔽，或被 FT 显式隔离 | 不下发普通操作；resume 不发给 dead rank |
| `exception` | 故障模式 | 推理过程中抛出异常 | `FT_pause` 会触发全局 pause；`FT_continue` 只记录事件 |
| `kill` | 故障模式 | 进程被强制终止，无异常上报，由 watchdog 发现 | `FT_pause` 进入 paused+dead；`FT_continue` 状态标 dead |
| `inactive_rank` | 独立输入 | 进程未退出也没有 exception，但 Mooncake 底层刷新 active ranks 后新增不可达 rank | 由 TokenizerManager 在 `update_active_ranks(ActiveRanksOutput)` 中观察 true->false |
| `ft_operation_in_progress` | 内部标记 | 控制面正在处理故障收敛、active mask 更新或 resume fanout | 防止并发 apply |
| `fault_handling_ready` | 派生条件 | `not ft_operation_in_progress and any(rank.state == paused)` | 只有此时允许 `apply` |
| `resume` | 内部动作 | 让 paused scheduler 退出 paused loop，恢复正常 event loop | retry/scale_down 共用 |

### 2.2 Rank 状态图

```text
+---------+       exception + FT_pause / inactive + FT_pause      +--------+
| HEALTHY | -----------------------------------------------------> | PAUSED |
+---------+                                                        +--------+
    |                                                                  |
    | kill / inactive / scale_down                                     | apply retry or scale_down
    v                                                                  v
+------+                                                         +---------+
| DEAD | <------------------------------------------------------- | HEALTHY |
+------+             scale_down marks rank dead                  +---------+

DEAD 不可逆；resume 不会发给 DEAD rank。
```

### 2.3 概念状态机与实现状态

文档可以用概念状态辅助理解：

| 概念状态 | rank 组合 | 普通请求 | apply |
|----------|-----------|----------|-------|
| 正常服务 | 全部 `healthy` | 允许 | 拒绝或 no-op |
| 故障处理中 | `ft_operation_in_progress=true` | `FT_pause` 下 503；`FT_continue` 按路由 | 409 |
| 等待 FT 决策 | 至少一个 `paused`，且无操作进行中 | 503 | 允许 |
| 降级服务 | `healthy + dead`，无 paused | 允许路由到 healthy | 拒绝，除非后续策略允许主动 scale_down |
| 不可服务 | 无 healthy/paused rank | 503 | 拒绝 |

实际代码不需要持久化这些实例状态。实现只需要：

```text
rank_states: list[healthy|paused|dead]
ft_operation_in_progress: bool
fault_handling_ready = !ft_operation_in_progress && any(paused)
```

### 2.4 状态转换表

| 当前 rank 组合 | 事件 | 模式 | 新 rank 组合 | 核心操作 |
|----------------|------|------|--------------|----------|
| all H | exception | noFT | fail-stop | 无 FT 控制面 |
| all H | kill | noFT | Mooncake 原生隔离或 fail-stop | 无 FT status |
| all H | exception | FT_pause | all P，或 P+D | 关闭 admission；pause healthy ranks；超时 rank 置 D |
| all H | kill | FT_pause | P+D | killed rank D；存活 rank P |
| P+D | apply retry | FT_pause | H+D | 用当前 dead mask 更新 active mask，resume paused |
| P+D | apply scale_down ranks=[x] | FT_pause | H+D' | 先把 x 标 D，再恢复 |
| all H | exception | FT_continue | all H | 上报事件，控制面不处理，不改 status |
| all H | kill | FT_continue | H+D 或不可服务 | 行为同 noFT；控制面只标 D |
| H | inactive_rank | FT_continue | H+D | status 标 D，不 pause |
| H | inactive_rank | FT_pause | P+D | status 标 D，pause remaining |
| H+D | exception | FT_pause | P+D | 连续故障场景；paused ranks 可通过 retry 恢复 |

---

## 3. 架构总览

### 3.1 进程/组件模型

```text
[HTTP Server / TokenizerManager 主进程]
├── FaultToleranceManager
│   ├── rank states: healthy / paused / dead
│   ├── ft_operation_in_progress
│   ├── status API
│   └── apply(retry|scale_down) 编排
│
├── Scheduler watchdog
│   └── kill: scheduler process exited -> FT event
│
├── DataParallelController (DP>1)
│   ├── self.status: TokenizerManager 转发来的 active DP ranks
│   ├── routing: 不向 inactive/dead rank 派发请求
│   └── kill event: DP>1 时上报 scheduler 子进程退出
│
└── Scheduler / ModelRunner
    ├── noFT normal event_loop
    ├── FT_pause: exception -> report -> paused loop
    ├── FT_continue: exception -> report -> no-op hook -> continue; control plane ignores status change
    ├── FT command: pause / apply_active_mask / resume
    └── Mooncake ElasticEP active_ranks / active_ranks_cpu
```

### 3.2 模块职责

| 组件 | 文件 | v5 职责 |
|------|------|---------|
| FT Controller | `python/sglang/srt/fault_tolerance/controller.py` | 维护 DP rank 状态、`ft_operation_in_progress`、API 校验 |
| FT Middleware | `python/sglang/srt/fault_tolerance/middleware.py` | `FT_pause` 等待决策/处理中拦截普通请求；放行 status/apply/health/metrics |
| HTTP API | `python/sglang/srt/entrypoints/http_server.py` | 注册 `status`、`apply` |
| TokenizerManager | `python/sglang/srt/managers/tokenizer_manager.py` | 编排 pause fanout、active mask 更新、resume fanout、watchdog 事件；截取 `ActiveRanksOutput` true->false 作为 inactive 输入 |
| DataParallelController | `python/sglang/srt/managers/data_parallel_controller.py` | 路由避开 inactive ranks；DP>1 时维护 scheduler process -> DP rank 映射并上报 kill |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | exception 策略、paused loop、resume、apply active mask |
| IPC schema | `python/sglang/srt/managers/io_struct.py` | FT command / output / fault event |
| Mooncake Elastic EP | `python/sglang/srt/elastic_ep/elastic_ep.py`, `mooncake.py` | active mask 生效与 inactive rank 观测 |

### 3.3 核心数据流

#### 3.3.1 `FT_pause + exception`

```text
rank k scheduler catches Exception
  -> rank k sends ExceptionFault(rank=k)
  -> rank k enters paused loop

TokenizerManager receives first exception
  -> ft_operation_in_progress = true
  -> admission closed
  -> fanout pause to all healthy ranks (idempotent)
  -> wait pause ack until timeout
      ack ranks -> paused
      timeout ranks -> dead
  -> if any dead exists, apply active_mask = ranks where state != dead
  -> ft_operation_in_progress = false
  -> apply is now allowed because paused ranks exist

POST /fault_tolerance/apply {instruction: "retry"}
  -> active_mask = ranks where state != dead
  -> apply active_mask if needed
  -> fanout resume to all paused ranks
  -> paused -> healthy; dead unchanged

POST /fault_tolerance/apply {instruction: "scale_down", params: {ranks: [...]}}
  -> mark params.ranks dead
  -> active_mask = ranks where state != dead
  -> apply active_mask
  -> fanout resume to all paused ranks
  -> paused -> healthy; dead unchanged
```

exception 会丢弃当前 scheduler batch、释放 KV 并返回 503；kill 不主动 abort、清理或重放 surviving ranks 的 in-flight 请求。推理中 kill 的受影响请求由原 engine/Mooncake 路径处理，其验收结果必须与 noFT 原生 Mooncake 一致。

#### 3.3.2 `FT_pause + kill`

```text
scheduler process exits
  -> watchdog detects exit
  -> killed rank -> dead
  -> alive healthy ranks -> paused
  -> fanout pause to alive ranks
  -> ft_operation_in_progress = false
  -> apply is allowed because paused ranks exist

apply retry:
  -> active_mask excludes current dead ranks
  -> resume paused ranks
  -> result: healthy + dead

apply scale_down ranks=[killed rank]:
  -> rank already dead, idempotent
  -> same result as retry
```

#### 3.3.3 `FT_continue + exception`

```text
scheduler catches Exception
  -> sends ExceptionFault(rank=k) for observability
  -> runs reserved no-op local hook
  -> continues normal event loop

TokenizerManager receives exception under strategy=continue
  -> logs/metrics only
  -> no rank state change
  -> no pause
  -> no active mask update
```

v5 初版不定义复杂本地 cleanup。hook 返回 false 或 hook 内再次异常时，scheduler 应 fail-stop，由 kill 路径处理。后续如果测试暴露 scheduler 状态残留 bug，只在该 hook 内补实际操作。

#### 3.3.4 `FT_continue + kill`

```text
scheduler process exits
  -> Mooncake/noFT native path handles peer loss as tested
  -> FT watchdog marks rank dead for status
  -> no pause fanout
  -> no automatic scale_down
```

#### 3.3.5 `inactive_rank` from ActiveRanksOutput

```text
Mooncake updates ElasticEP active_ranks
  -> scheduler emits ActiveRanksOutput(status=[...])
  -> TokenizerManager.update_active_ranks(status=[...])
  -> FT controller compares previous active mask vs new status
      newly false ranks = inactive_ranks
  -> TokenizerManager forwards ActiveRanksOutput to DPC
  -> DPC self.status refreshes for routing only

FT_continue:
  -> mark inactive ranks dead in FT status
  -> do not pause remaining ranks

FT_pause:
  -> mark inactive ranks dead
  -> pause remaining healthy ranks
  -> apply retry can resume paused ranks with current dead mask
```

---

## 4. 外部接口规范

### 4.1 `GET /fault_tolerance/status`

**用途：** 返回 FT 控制面视角的 DP rank 状态。

**最小响应 200：**

```json
{
  "ranks": [
    {"rank": 0, "state": "paused"},
    {"rank": 1, "state": "dead"},
    {"rank": 2, "state": "paused"}
  ]
}
```

**字段约束：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `ranks` | array | 长度等于 DP size |
| `ranks[].rank` | int | DP rank id |
| `ranks[].state` | string | 只允许 `healthy|paused|dead` |

实现可以在 debug 响应或 metrics 中暴露 `mode`、`ft_operation_in_progress`、last fault message，但 v5 对外契约只要求 rank 三态。

### 4.2 `POST /fault_tolerance/apply`

**用途：** 执行 `retry` 或 `scale_down`。

**请求：**

```json
{
  "fault_tolerance_instruction": "scale_down",
  "fault_tolerance_timeout": 60,
  "fault_tolerance_params": {
    "ranks": [1]
  }
}
```

`retry` 请求不需要参数：

```json
{
  "fault_tolerance_instruction": "retry"
}
```

**成功 200：**

```json
{
  "success": true,
  "message": "fault tolerance recovery succeeded",
  "ranks": [
    {"rank": 0, "state": "healthy"},
    {"rank": 1, "state": "dead"},
    {"rank": 2, "state": "healthy"}
  ]
}
```

**请求字段：**

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `fault_tolerance_instruction` | string | 是 | `retry|scale_down` |
| `fault_tolerance_timeout` | int | 否 | > 0，默认使用 server args |
| `fault_tolerance_params.ranks` | int[] | scale_down 必填 | DP rank，非空，不能隔离全部 remaining ranks |

**通用前置条件：**

- FT 已开启。
- `ft_operation_in_progress=false`。
- 当前存在 `paused` rank。否则说明引擎不处于等待 FT 请求的故障处理态，返回 400。

**instruction: retry**

- 不接受参数。
- 不新增 dead rank。
- 使用当前 status 中 `state=dead` 的 ranks 生成 active mask。
- 对所有 paused ranks 下发 resume。

**instruction: scale_down**

- 先把参数中的 ranks 标为 dead。
- 再走与 retry 相同的恢复动作。
- 参数中已经 dead 的 rank 幂等接受。

**失败语义：**

| 场景 | Code | 说明 |
|------|------|------|
| FT 未开启 | 503 | FT API shim 返回 `fault_tolerance_disabled`，不暴露 disabled status body |
| `ft_operation_in_progress=true` | 409 | 故障仍在收敛，暂不接受 apply |
| 无 paused rank | 400 | 引擎不处于等待 FT 请求的故障处理态 |
| unknown instruction | 400 | 不支持 `pause` |
| scale_down ranks 非法 | 400 | 空、越界、隔离全部 |
| 非 Mooncake + FT enabled | 启动拒绝 | 初版不支持非 Mooncake 半功能模式 |
| resume/apply mask timeout | fail-stop | 执行阶段失败不返回 500，避免半恢复继续服务 |

---

## 5. 行为矩阵

### 5.1 主矩阵

| 模式 | exception | kill | inactive_rank |
|------|-----------|------|---------------|
| noFT | 无法处理，fail-stop | Mooncake 原生隔离；无 FT status | Mooncake/DPC 原生 routing，FT 不展示 |
| FT_pause | 全局 pause；超时 rank dead；等待 apply | killed rank dead，存活 rank paused；retry/scale_down 均可恢复 | 新 inactive rank dead，主动 pause remaining |
| FT_continue | 上报异常但控制面不处理，不改 status | 与 noFT 一样继续；FT status 标 dead | status 标 dead；不 pause |

### 5.2 apply 矩阵

| rank 组合 | apply retry | apply scale_down |
|-----------|-------------|------------------|
| all healthy | 400，无 paused rank | 400，初版不支持主动 scale_down |
| paused only | resume paused -> healthy | 先标参数 ranks dead，再 resume remaining paused |
| paused + dead | 使用当前 dead mask，resume paused -> healthy+dead | 可再标更多 dead，resume remaining paused |
| healthy + dead，无 paused | 400，无 paused rank | 400，初版不支持运行中主动隔离 |
| 操作进行中 | 409 | 409 |

---

## 6. 核心机制

### 6.1 Exception 处理

`FT_pause` 下，exception 不是单 rank 局部恢复事件。任一 rank 上报 exception 后，控制面必须：

1. 设置 `ft_operation_in_progress=true`。
2. 关闭 `FT_pause` 普通请求 admission。
3. 对所有 still-healthy ranks 下发 pause，幂等处理已经 paused 的 rank。
4. 等待 pause ack；超时未 ack 的 rank 标 dead。
5. 如存在 dead rank，更新 Mooncake active mask。
6. 设置 `ft_operation_in_progress=false`，进入等待 apply 状态。

`FT_continue` 下，scheduler 仍上报 exception 事件，但控制面不改变 status。

### 6.2 Kill 处理

kill 没有异常上报，由进程退出检测触发。

`FT_pause`：

- killed rank -> `dead`。
- alive healthy ranks -> `paused`。
- `apply retry` 和 `apply scale_down ranks=[killed]` 均可恢复 remaining paused ranks。

`FT_continue`：

- killed rank -> `dead` for status。
- 不 pause alive ranks。
- 自动更新 routing / active mask 避开 dead rank；不发 pause，不等待用户 apply。

### 6.3 Mooncake inactive rank 处理

`ActiveRanksOutput.status` 是 Mooncake active-rank 刷新的控制面可见结果。当前链路是 `Scheduler -> TokenizerManager.update_active_ranks() -> DPC.update_active_ranks()`，因此 inactive_rank 必须在 TokenizerManager 入口截取 true->false 后更新 FT controller，再继续转发给 DPC 做路由更新。

DPC 的 `self.status` 只作为路由状态，不作为 FT controller 的 inactive 状态源，也不反向上报 inactive_rank：

- `FT_continue`：新增 inactive rank 标 dead，remaining 继续。
- `FT_pause`：新增 inactive rank 标 dead，并 pause remaining healthy ranks。
- false->true 表示底层 active-rank 路径观测到 rank rejoin。FT controller 可以把对应 rank 从 `dead` 恢复为 `healthy`，但该恢复只接受 Mooncake/ElasticEP 已重新确认 active 的 rank，不由 `retry` 或普通 `apply` 随意复活。

### 6.4 通用恢复动作

```text
recover():
  # 两阶段提交：先执行容错动作，收集结果，再更新对外 status。
  pending_scale_down = params.ranks if instruction == scale_down else []
  active_mask = [rank.state != dead and rank not in pending_scale_down for rank in ranks]
  apply active_mask to live schedulers / Mooncake
  if not shutdown-live-rank enabled:
    resume pending live scale_down ranks so they leave paused loop but receive no normal traffic
  resume remaining paused ranks
  update DPC routing mask
  if shutdown-live-rank enabled:
    DPC-local shutdown pending live scale_down ranks after mask/routing,
    and suppress duplicate watchdog kill event by treating the exit as intentional
  commit public FT status: pending_scale_down -> dead, paused -> healthy
```

`retry`：

```text
recover()
```

`scale_down`：

```text
recover()
```

`scale_down` 对 live paused rank 默认走 shutdown 路径：先让所有 live scheduler 应用 active mask，再 resume surviving ranks，发布 DPC routing inactive，最后由 DPC 本地终止被隔离 scheduler 并 ack。默认 shutdown 的原因是让被逻辑隔离的 rank 不再作为 parked/idle scheduler 参与后续 topology change，避免后续连续 scale_down 或 rejoin 时还要依赖一个已不应接收普通流量的进程。实现保留 `SGLANG_FT_SCALE_DOWN_SHUTDOWN_LIVE_RANK=0` 兼容 non-shutdown 路径；该路径下被隔离 rank 物理进程仍存在，需接收 inactive active-mask，并在 paused loop 中收到 `resume` 后退出到不会接收普通请求的 idle/control 状态。

### 6.5 Admission 与路由

- `FT_pause` 中，只要存在 paused rank 或 `ft_operation_in_progress=true`，普通请求返回 503。
- `FT_continue` 不使用全局 admission 关闭；路由必须避开 dead/inactive ranks。
- 如果请求显式指定 `routed_dp_rank` 且该 rank 非 healthy，必须拒绝。
- `/fault_tolerance/status`、`/fault_tolerance/apply`、`/health`、`/ping`、`/metrics` 不被 admission gate 拦截。
- 生成型健康检查不属于 FT 管理面 bypass。如果 `/health_generate` 实际会进入 generate/scheduler 路径，则在 `FT_pause` 下按普通请求处理，返回 503 或由部署侧停用。

### 6.6 请求清理边界

exception 与 kill 的请求边界不同：

- exception：丢弃当前 scheduler batch，释放 KV cache，向 tokenizer/client 返回 503；不做请求重放。
- kill：不主动 abort surviving ranks 的 in-flight requests，不主动清空 surviving scheduler running batch。
- 推理过程中 kill 的受影响请求表现必须与原生 Mooncake noFT 行为一致：如果原生 Mooncake 在请求是否路由到被 kill rank 时都能返回 200，FT 必须保持；反之不要求更强。
- Mooncake/model runner 内部发现 rank 状态变化时，按原有 engine 路径处理请求结果。

paused loop 只阻止后续普通 work 继续进入正常 event loop；已经在底层执行的请求由原路径收敛。

---

## 7. 实现 Guardrails

本节来自 v3/v4 文档和 SGLang FT 原型提交中的踩坑经验。它们不改变 v5.1 的简化语义，只约束实现不要重新踩已知问题。

### 7.1 FT disabled 必须物理隔离

带 FT 代码但 `--enable-fault-tolerance=false` 时，必须保持 noFT 行为：

- 不安装 FT admission gate。
- 不启动 FT watchdog callback。
- 不拦截 DPC routing。
- 不实例化、不启动 FT 状态机。
- `status` / `apply` 可以保留为薄 API shim，但必须统一返回 HTTP 503 和 `fault_tolerance_disabled`；不得返回全 healthy 或 disabled status body。
- kill 场景必须与无 FT 代码时的 Mooncake 原生隔离行为一致。

### 7.2 Command fanout 约束

- FT command 必须带 `request_id` 并等待 ack 聚合。
- DP-attention 非 local-control broadcast 路径中，FT command 必须从 DPC control root 进入，再由 scheduler `recv_requests()` 广播；不能按 `target_ranks` 直接单播到非 root rank，否则 target rank 可能因为广播源没有 control request 而收不到命令。scheduler 收到广播后再按 `target_ranks` 过滤是否执行。
- `pause` 只发给 live healthy ranks。
- `resume` 只发给 paused ranks。
- `apply_active_mask` 只发给 live ranks。
- 任何 FT command 都不得发给真实已退出的 `dead` 或 DPC status 已 inactive 的 rank。
- 例外：本次 `scale_down` 中即将被隔离、但物理进程仍 live 且 DPC routing 尚未提交 inactive 的 rank，必须允许接收 `apply_active_mask`；non-shutdown 路径还必须接收 `resume` 以退出 paused loop。
- shutdown 路径不是普通 scheduler command fanout：在 active mask 和 routing 更新完成后由 DPC 本地终止目标 scheduler，并向 tokenizer 返回 shutdown ack。
- 执行阶段 ack timeout、`success=false`、部分成功，初版统一 fail-stop。

### 7.3 DPC routing 竞态

DPC `self.status` 更新后，不能长期依赖 `sleep` 避免新请求 race。v5 初版应满足以下之一：

- routing update 有 ack/barrier，确认 DPC 已停止向 dead/inactive rank 派发。
- 或在测试中覆盖“status 更新后立即 generate”，证明不会打到 dead/inactive rank。

当前实现中，`scale_down` 在 resume ack 后才发布 `ActiveRanksOutput` 给 DPC；由于这条 ZMQ 控制消息没有 DPC ack，发布后短暂 `sleep(1)` 是临时传播等待，避免 apply 已返回 200 而紧随其后的 generate 仍按旧 routing 打到被隔离 rank。该 sleep 是兼容措施，不是长期正确性边界；后续应替换为显式 routing ack/barrier。

如果所有 DPC status 均为 false，DPC 必须拒绝请求或触发不可服务状态，不能在 round-robin 中空转。

### 7.4 active mask 单一入口

所有 active mask 更新必须通过统一 helper，例如 `apply_active_rank_mask(mask)`。该 helper 负责：

- 更新 GPU 侧 `active_ranks`。
- 同步 CPU mirror `active_ranks_cpu`。
- 刷新 Mooncake EP members / buffer 成员。
- 触发必要的 EPLB/dispatcher 后置同步。

禁止 TokenizerManager、Scheduler、Mooncake dispatcher 各自手写 active mask 更新逻辑。

native DP 启动中，每个 scheduler 可能只有一元素本地 ElasticEP state，而 FT 控制面携带的是全局 DP-rank mask。此时 helper 可以把全局 DP mask 在本地折叠为 `[true]`，保持本 scheduler 的一元素 Mooncake mirror 可用；真正的 DP 级隔离由 DPC routing 和控制面状态负责。

### 7.5 fallback gate

v5 主路径是 `active_mask + resume`。如果实测 Mooncake 仍需要 v3/v4 的 `reinit`、`post_recovery_smoke`，只能作为隐藏 fallback。`park_idle/resume` 不是对外 API，也不改变 rank 状态语义；它作为 topology-change 后的内部空闲安全点存在，用于让 Mooncake/DP-attention/EPLB 在“一个响应完成”和“下一次 forward”之间处理 active-mask 变化。

- 不改变对外 API 和状态语义。
- 不重新引入 `dead_isolation_applied`、复杂 instance state、retry dead reason 判断。
- 启用或删除 fallback 前，必须通过连续 scale_down、final response 后首个请求、inactive rank、FT disabled kill parity 测试。

### 7.6 命名约束

不要把轻量恢复探测命名为 `health_check`，除非它确实验证了 remaining ranks 的通信健康。若只是恢复后的烟测，命名为 `post_recovery_smoke`。

### 7.7 启动配置统一入口

FT 支持性判断必须集中到一个 helper，例如：

```python
is_ft_supported_config(server_args) -> tuple[bool, str]
```

该 helper 至少统一判断：

- 实际模型计算拓扑 effective `tp_size == 1`。该值必须来自 SGLang 参数归一化后的 topology / helper 字段，不能用 raw CLI `--tp-size` 直接判断；raw `--tp-size>1` 在 MoE/Mooncake 中可能只是 EP 或全局 world 维度。
- `pp_size == 1`
- 单机部署
- Mooncake active-rank path 是否可用
- PD/disaggregation、NPU、Ray engine、`tokenizer_worker_num > 1` 等初版 unsupported 组合

DP attention 不在该拒绝列表中。FT 对 DP attention 透明通过，但必须有默认启动参数下的 smoke/回归测试覆盖 status、pause、apply retry、apply scale_down。

避免 v4 中 `moe_a2a_backend`、`elastic_ep_backend` 多处判断不一致的问题。

---

## 8. 边界条件

| 边界 | 行为 |
|------|------|
| `apply retry` 无 paused rank | 400 |
| `FT_pause + kill` 后 `apply retry` | 允许，按当前 dead mask 恢复 |
| `apply scale_down` 空 ranks | 400 |
| `apply scale_down` unknown rank | 400 |
| `apply scale_down` 隔离全部 remaining ranks | 400 |
| `apply scale_down` 包含已 dead rank | 幂等接受 |
| pause fanout 超时 | 超时 rank -> dead，继续收敛 |
| resume fanout 超时 | 初版 fail-stop |
| DPC status 全 false | 无可服务 rank；普通请求 503 |
| recovery 中新增 kill | 初版 fail-stop |
| FT 关闭但代码存在 | 必须保持 noFT 行为 |

---

## 9. 测试计划

### 9.1 P0 用例

| ID | 场景 | 期望 |
|----|------|------|
| TC-v5-001 | noFT + kill | FT API 503 disabled；Mooncake 原生隔离；后续 generate 200 |
| TC-v5-002 | FT_continue + exception | faulting request 503 discard；status 不变；不 pause；后续 generate 200 |
| TC-v5-003 | FT_continue + kill | static 与 in-flight kill 均不 pause；rank dead；routing/active mask 避开 dead rank；后续 generate 200 |
| TC-v5-004 | FT_pause + exception | faulting request 503 discard；全部 paused；apply retry 全部 healthy；apply scale_down 隔离指定 rank；后续 generate 200 |
| TC-v5-005 | FT_pause + kill | static 与 in-flight kill 均进入 killed dead + surviving paused；retry 或 scale_down 后 remaining generate 200 |

P0 只覆盖正常路径架构能力。pause 场景的 retry 与 scale_down 是同一核心场景的执行变体，因此脚本数量可以多于场景数量。推理中 kill 的受影响请求结果必须先用 noFT 原生 Mooncake probe 确认，再把 FT 用例收紧为完全一致。

### 9.2 P1/P2 用例

| ID | 场景 | 期望 |
|----|------|------|
| TC-v5-015 | apply 在 `ft_operation_in_progress=true` 时到达 | 409 |
| TC-v5-016 | scale_down 空数组、unknown rank、全集 | 400 |
| TC-v5-017 | 显式 routed_dp_rank 指向 dead/paused | 拒绝，不派发 |
| TC-v5-018 | routing status 更新后立即 generate | 不派发到 dead/inactive rank |
| TC-v5-019 | DPC status 全 false | 不空转，明确拒绝或不可服务 |
| TC-v5-020 | 连续故障：H+D 后 exception -> P+D -> retry | remaining 恢复为 H+D |
| TC-v5-021 | false->true inactive mask refresh / rejoin | Mooncake/ElasticEP 已确认 active 后，FT dead rank 可恢复为 healthy；不能由 retry/apply 任意复活 |
| TC-v5-022 | 非 Mooncake + FT enabled | 启动拒绝 |
| TC-v5-023 | actual effective TP>1 + FT | 启动拒绝；raw `--tp-size>1` 不能作为唯一判断依据，MoE/Mooncake 合法形态不得误拒 |
| TC-v5-024 | FT_pause + exception，某 rank pause timeout | timeout rank dead；apply retry 后 remaining healthy+dead |
| TC-v5-025 | FT_continue + inactive_rank | TokenizerManager 收到 `ActiveRanksOutput` 新 false 后 FT status 标 dead |
| TC-v5-026 | FT_pause + inactive_rank | 新 false rank dead，remaining paused；apply retry resume |
| TC-v5-027 | command target filtering | true dead / DPC inactive rank 不执行 FT command；pending live scale_down rank 可收 active_mask，non-shutdown 路径可收 resume |
| TC-v5-028 | active mask helper 同步 CPU/GPU mirror | `active_ranks` 与 `active_ranks_cpu` 一致，EP members 刷新 |
| TC-v5-029 | live rank scale_down shutdown path | 默认 shutdown，不重复触发 watchdog kill；`SGLANG_FT_SCALE_DOWN_SHUTDOWN_LIVE_RANK=0` 覆盖 non-shutdown 兼容路径 |

---

## 10. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v5.1.5 | 2026-06-30 | 对齐当前实现：raw `--tp-size` 不等同 actual effective TP、跨节点 FT 拒绝、false->true 作为 rejoin、默认 live scale_down shutdown、补充 park_idle、routing sleep、native DP mask fold 说明 |
| v5.1.4 | 2026-06-18 | 修正 inactive_rank 链路：TokenizerManager 截取 `ActiveRanksOutput`；DPC `self.status` 仅用于路由，DPC 反向上报只保留 kill |
| v5.1.3 | 2026-06-18 | 定案 FT disabled API 为 503 unavailable、非 Mooncake FT 启动拒绝、执行阶段失败统一 fail-stop |
| v5.1.2 | 2026-06-18 | 明确 health_generate 不 bypass、DP attention 透明通过、FT_continue exception no-op hook、false->true 归入未来 rejoin |
| v5.1.1 | 2026-06-18 | 补充 v3/v4 踩坑 guardrails：FT disabled 隔离、fanout target、DPC routing 竞态、active mask helper、fallback gate |
| v5.1.0 | 2026-06-18 | 根据讨论结论简化：只保留 apply；删除 dead_isolation_applied；实例状态改为内部 operation flag；允许 FT_pause+kill retry |
| v5.0.0 | 2026-06-18 | 根据真实故障处理策略重写 v4：exception/kill/inactive_rank、status/retry/scale_down、noFT/FT_pause/FT_continue |
