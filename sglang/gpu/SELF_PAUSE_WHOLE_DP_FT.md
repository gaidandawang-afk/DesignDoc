# SGLang Scheduler 自暂停与 whole-DP 容错设计

状态：实现与用例依据（normative）  
适用范围：DP attention、Mooncake elastic EP、EPLB、`PP=1`

本文覆盖新的最小容错架构。与同目录旧 `README.md` 冲突时，以本文为准。

## 1. 核心约束

1. Mooncake 的 `active_ranks` 是数据面观察量，不是 pause 策略下的控制面拓扑真相。
2. Scheduler 在 forward 异常边界自行暂停；中心不发送 pause，也不等待 pause ACK。
3. 原 watchdog 完整保留。`process_alive_mask` 是进程存活的权威来源。
4. retry 只验证 recoverable exception；kill 或进程退出只能验证 scale-down。
5. scale-down 总是关闭整个目标 DP：目标 DP 内全部 `A*C` 个 Scheduler 都被 kill。
6. rejoin 总是按原生 `nnodes` node process group 重拉，不引入 DP-scoped rejoin。
7. recover 只提交 `expected_dp_mask` 和 DPC route，不向 Scheduler 下发命令。
8. 公共状态只有 `healthy`、`unhealthy`、`dead`、`disabled`，没有 `paused`。
9. 不实现非预期异常窗口的健壮性分支；重构后的生产代码必须少于旧实现。

## 2. 拓扑与状态

当前只支持 `PP=1`。设：

```text
T = tp_size
D = dp_size
R = T / D = A * C
global scheduler ranks = [0, T)
members(dp) = [dp * R, (dp + 1) * R)
```

控制面只维护：

```text
expected_dp_mask: bool[D]
process_alive_mask: bool[T]
disabled_dp_ranks: set[dp_rank]
unhealthy_dp_ranks: set[dp_rank]
ft_operation_in_progress: bool
route_mask: bool[D]                 # manager 私有的最后 DPC route
watchdog_leases: node -> (time, global scheduler ranks)
```

含义：

- `expected_dp_mask` 是已提交执行拓扑。初始全 `true`；scale-down 清零，recover 置回。
- `process_alive_mask` 是 global Scheduler 进程粒度，而不是 node 或 DP 粒度。
- 一个 DP 只有在其全部 global Scheduler 成员存活时才是 process-alive。
- `disabled_dp_ranks` 只表示：被 scale-down 的 DP 已完成原生 rejoin 数据面恢复，等待 recover。
- `unhealthy_dp_ranks` 只记录 pause 策略下 Scheduler 抛出的 recoverable exception。
- 不保存 Mooncake active mask 的控制面副本，不保存 provisional、incarnation 或 generation。

FT 启用配置必须满足已有 Mooncake/DP/`PP=1` 等约束，并要求：

- `enable_dp_attention=true`
- `enable_eplb=true`
- `tp_size % (dp_size * attn_cp_size) == 0`

## 3. watchdog 与 process_alive_mask

修改前的 watchdog 机制必须保留：

1. 每个 DPC 用 `SubprocessWatchdog` 监控自己启动的 Scheduler children。
2. DPC 在 watchdog 线程启动前发送初始 heartbeat，之后周期发送 heartbeat。
3. child 退出时立即发送 `ProcessActiveRanksOutput(active=false)`。
4. recover rejoiner 启动完成后发送 `ProcessActiveRanksOutput(active=true)`。
5. Node0 manager 保存 node lease；heartbeat 超时后把该 node 拥有的 global ranks 置为 down。
6. shutdown 完成条件直接复用上述 ProcessDown 事实，不增加 shutdown ACK 协议。

heartbeat 和 ProcessUp/Down 中的 rank 改为 global Scheduler rank。这样 `A*C>1` 时，
外部只 kill 一个成员也能先把整个 DP 投影为 `dead`，随后 scale-down 再 kill 其余 sibling。

heartbeat 不把 false 位恢复为 true；只有显式 ProcessUp 可以恢复进程位。没有
incarnation、sequence、retired generation、owner pin 或 lease 补 ACK。

## 4. 公共状态

状态按以下优先级计算：

| 条件 | state |
| --- | --- |
| DP 任一成员进程不存活 | `dead` |
| `expected=false`，全部成员存活且原生 active 已恢复 | `disabled` |
| `expected=true` 且收到 Scheduler exception | `unhealthy` |
| 其他 | `healthy` |

`process_alive_mask` 只决定进程事实。Mooncake active 仅用于确认 rejoin 数据面已恢复，
以及 continue 策略的路由，不可把 pause 策略下的 expected 拓扑改写掉。

被 scale-down 的 DP 后续退出不会形成新 incident，因为 incident 只检查 expected DP。
本设计不额外处理 disabled rejoiner 在 recover 前再次异常等非预期窗口。

## 5. pause 策略

### 5.1 recoverable exception

Scheduler event loop 捕获 forward 异常后：

1. 丢弃当前 in-flight window；
2. 设置本地 `_engine_paused=true` 和本地 fail-stop deadline；
3. 上报 `FaultToleranceRankFaultOutput`；
4. paused loop 继续处理控制消息，但不再执行推理。

中心收到异常只记录 `unhealthy` 并关闭请求入口。

### 5.2 进程退出

watchdog 先更新 `process_alive_mask`，中心据此关闭请求入口。其他 Scheduler 在一次
forward 后观察到 `last_active_ranks & ~active_ranks`，在 EPLB 和第二次 forward 前
抛异常，进入与 5.1 相同的自暂停流程。中心不发送 pause。

## 6. retry

retry 的严格语义是“集群没有真实故障，恢复异常前已提交的数据面状态”。

前置条件：

- 存在 `unhealthy`；
- 所有 `expected=true` DP 的全部进程仍存活。

流程：

```text
send retry to expected DP schedulers
  -> active_ranks.copy_(last_active_ranks)
  -> active_ranks_cpu.copy_(last_active_ranks.cpu())
  -> clear local pause and deadline
publish DPC route = expected_dp_mask
clear unhealthy
```

retry 不携带 active mask，不接受 Mooncake 上报，不触发 EPLB。控制命令完成仍使用原有
Scheduler command response；不增加全局 pause barrier。

若已完成 `4 -> 3` scale-down，`expected_dp_mask` 已是三实例。之后 retry 只能恢复这
三个实例，不能因 runtime active 上报重新扩成四实例。

进程 kill、真实进程故障和错误状态下调用 retry 均不属于支持场景。

## 7. scale-down

scale-down 可指定任意非空 expected DP 集合，但不能移除全部 expected DP。调用时必须
已有 exception 或 process-down incident。

单阶段流程：

```text
candidate = expected_dp_mask - requested_dp
send shutdown(requested_dp) to owner DPC nodes
  -> each DPC kills every local Scheduler whose dp_rank is requested
wait until original watchdog marks every requested global rank down
send scale_down(expand(candidate)) to survivor DP schedulers
  -> install sparse global active mask
  -> force EPLBManager.rebalance()
  -> snapshot active mask to last_active_ranks
  -> clear local pause and deadline
publish DPC route = candidate
commit expected_dp_mask = candidate
clear unhealthy
```

只有一个新增的 DPC control endpoint 用于发送 kill 请求。没有 prepare/rebalance/resume
三阶段命令，没有 shutdown ACK、grace period 或 lease 代 ACK。

外部可以先 kill 任意 Scheduler，再对该失活 global rank 所属 DP 调用 scale-down。
scale-down 必须继续 kill 该 DP 的其余 sibling。完成后该 DP 状态保持 `dead`。

## 8. rejoin 与 recover

用例和运维必须 kill、重拉需要 rejoin 的 DP 所在 `nnode` 的完整进程组；不新增
DP-scoped respawn。

顺序：

```text
whole node process group rejoin
ProcessUp -> process_alive_mask 恢复，但 expected=false，状态仍 dead
native forward -> Mooncake 恢复 active + 原生 EPLB/第二次 forward
native active 与 process-alive 都完整 -> 状态 disabled
recover(dp)
  -> expected_dp_mask[dp] = true
  -> publish DPC route
  -> 状态 healthy
```

recover 不发送 Scheduler 命令，不重排专家，不做 topology expansion，也没有
`recover_commit` 或 provisional 状态。所有数据面扩容必须在 recover 前由原生 rejoin
forward 完成。

## 9. continue 策略

continue 保持 no-FT 风格：

- membership loss 继续走 Mooncake 原生 EPLB 和第二次 forward；
- DPC route 取 `expected & process_alive & native_active`；
- ProcessUp 本身不开放 route，必须等待新的 native active 上报；
- recoverable exception 丢弃当前请求后继续 event loop，不建立 pause incident，公共
  状态保持 `healthy`，也不执行 retry reset。

## 10. 明确删除或不实现

必须删除：

- `mooncake_active_ranks`、`process_active_ranks` 等旧 DP 粒度权威 mask；
- `paused_dp_ranks`、`PAUSED`、中心 pause 命令及 pause ACK；
- provisional active、`recover_commit`；
- 两阶段或三阶段 scale-down；
- incarnation、sequence、retired generation、owner pin；
- shutdown ACK grace、lease-DOWN 补 ACK。

不要求处理：

- 真实故障后错误调用 retry；
- FT 操作中消息丢失、DPC 换代或重复请求；
- rejoiner 在 disabled/recover 窗口再次退出；
- 操作中再次发生独立故障。

## 11. 最小验收用例

1. exception -> `unhealthy` -> maskless retry -> `healthy`，无 EPLB。
2. kill 一个 Scheduler -> `dead` -> scale-down，目标整个 DP 退出。
3. `A*C>1` 时 kill 一个 sibling，scale-down 后同 DP 全部 sibling 退出。
4. `4 -> 3` 后 exception retry，始终保留三实例且被移除 DP 仍不可路由。
5. scale-down 后按完整 nnode rejoin：`dead -> disabled -> recover -> healthy`。
6. continue kill 保留原生 EPLB、第二次 forward 和 route 行为。
7. watchdog child exit、heartbeat lease timeout、ProcessUp 和进程粒度投影单测。

kill 用例不得替代 retry 用例；retry 只能用 exception 注入验证。

## 12. 代码量门槛

本次重构删除的机制多于新增机制。生产代码相对重构基线必须净减少；若净增加，视为
架构实现错误，必须先删除冗余状态、协议和防御分支，不能以“健壮性”为理由接受。
