# SGLang FT / Elastic EP 工程实现总结（面向开发工程师）

> 版本：v0.1  
> 日期：2026-06-27  
> 目标读者：SGLang 后端开发、推理框架开发、SRE、测试工程师  
> 对齐基线：
> - 本地实现仓：`D:\Codex\repos\sglang`，分支 `codex/v5-native-sglang-bugfix`，提交 `b4068cf5a96a7b4ab763663646c359c1373976d8`
> - 设计文档：`designs/v5/Spec/sglang_fault_tolerance_design_v5_arch_zh.md`
> - 详细实现设计：`designs/v5/Spec/sglang_fault_tolerance_design_v5_noft_impl_zh.md`
> - 发布前测试计划：`designs/v5/0624/SGLang Elastic EP 与 FT v5 发布前功能测试计划.md`
> - 社区目标：[`sgl-project/sglang#8961`](https://github.com/sgl-project/sglang/pull/8961)

## 1. 一页结论

如果只看一句话，当前这套 FT 代码已经不是“纯设计稿”，而是一个真实落在 SGLang 控制面里的 v5 实现：

- 它已经把 FT 作为一套显式控制面能力接进了 `TokenizerManager`、`Scheduler`、`DataParallelController` 和 HTTP server。
- 它已经实现了 `status/apply` 管理接口、`pause/continue` 两种异常策略、`retry/scale_down` 两种恢复动作，以及基于 `ActiveRanksOutput` 的 inactive-rank 感知。
- 它不是单独替代 Elastic EP，而是叠加在 Mooncake Elastic EP 之上，通过 active-rank mask、DPC routing 和 scheduler 控制命令来做收敛。
- 它和社区最终合入主线的路径已经部分重合，但并不完全同构。社区主线把 `#8961` 拆成了 M1~M6 六个 PR 分阶段合入；当前本地分支额外引入了专门的 FT 控制面目录、HTTP API 和更强的外部语义。

因此，这份总结最重要的判断是：

1. 当前代码已经具备“工程上可运行的 FT 骨架”，不是纸面方案。
2. 当前代码的“实现了”与“已经被充分验证”不是一回事，验证面明显落后于实现面。
3. 如果目标是“最终对齐社区合入形态”，主要 GAP 已经不在核心算法，而在接口形态、文档一致性、支持矩阵边界和测试完备性。

## 2. 为什么要这样设计

`#8961` 的原始动机非常清楚：MoE 推理在大规模部署时，既要能在 rank/GPU 失效后继续服务，也要能在恢复后重新拿回吞吐。它最初的 roadmap 可以压缩成三件事：

1. 先让 EP/dispatch 层能“屏蔽掉坏 rank 继续跑”。
2. 再让 scheduler / DP routing 能感知故障，避免请求继续打到坏 rank。
3. 最后补齐弹性动作，包括缩容、补回失效 rank、恢复吞吐。

当前本地 v5 FT 实现延续了这个方向，但做了一个更工程化的取舍：把“故障观察”和“恢复决策”显式上收为控制面能力，而不是完全隐在底层通信后端里。

这么做的原因有三个：

- 对外语义更稳定。上层看到的是 `healthy/paused/dead` 三态和 `retry/scale_down` 两种动作，而不是一堆 Mooncake/dispatcher 内部状态。
- 调度链路更可控。故障不只影响 EP，实际上还会影响 admission、routed_dp_rank、DPC 派发和 scheduler event loop，这些都需要统一编排。
- 更容易把“继续服务”和“人工介入恢复”分开。`continue` 策略解决的是“先别把集群全停掉”，`pause + apply` 解决的是“让控制面做一次显式收敛”。

## 3. 当前代码的实现形态

### 3.1 控制面骨架

当前代码里已经存在专门的 FT 控制面目录：

- `python/sglang/srt/fault_tolerance/controller.py`
- `python/sglang/srt/fault_tolerance/middleware.py`

其中 `FaultToleranceManager` 维护 rank 三态和 `ft_operation_in_progress`，并负责：

- status 序列化
- `retry/scale_down` 参数校验
- `kill/exception/inactive` 状态收敛
- `active_mask`、`resume_targets`、`pending_scale_down_ranks` 计算

这意味着当前实现不是“借用 pause_generation 拼一套 FT 语义”，而是已经形成了独立的控制面状态机。

### 3.2 外部接口

当前 HTTP server 已经暴露两条 FT 管理接口：

- `GET /fault_tolerance/status`
- `POST /fault_tolerance/apply`

CLI 也已经暴露三项入口参数：

- `--enable-fault-tolerance`
- `--fault-tolerance-on-error-strategy pause|continue`
- `--fault-tolerance-timeout`

同时，HTTP middleware 已经把 FT admission gate 接到了请求入口：

- `pause` 策略下，只要正在处理故障或仍有 paused rank，普通请求返回 `503`
- `/fault_tolerance/status`、`/fault_tolerance/apply`、`/health`、`/metrics`、`/ping` 不被 gate 拦截

### 3.3 故障事件从哪里来

当前实现有三类故障输入：

1. `exception`
2. `kill`
3. `inactive_rank`

它们的来源分别是：

- `exception`：`Scheduler._run_event_loop_fault_tolerance()` 捕获 event loop 异常，上报 `FaultToleranceRankFaultOutput(fault_type="exception")`
- `kill`：`DataParallelController` 里的 `SubprocessWatchdog` 监听 scheduler 进程退出，上报 `FaultToleranceRankFaultOutput(fault_type="kill")`
- `inactive_rank`：scheduler 周期性上报 `ActiveRanksOutput`，`TokenizerManager.update_active_ranks()` 先送进 FT controller，再继续发给 DPC

这条链路说明当前 FT 不是只处理“进程死掉”，也能处理“进程还活着，但底层 active-rank mask 已经把它摘掉”的场景。

### 3.4 恢复动作怎么做

`apply` 当前只支持两种 instruction：

- `retry`
- `scale_down`

两者共用同一套恢复路径：

1. 计算目标 `active_mask`
2. 必要时先 `park_idle`
3. 下发 `apply_active_mask`
4. 下发 `resume`
5. 更新 DPC routing
6. 可选地对 live scale-down target 再做本地 `shutdown`
7. 最后提交外部可见状态

这套顺序的核心价值是：先收敛真实运行态，再提交 public status，避免“状态先改了，但底层还没稳定”的半恢复窗口。

## 4. 当前代码的实际表现

这里的“实际表现”分两层看：

- 第一层是当前代码定义出来的外部行为。
- 第二层是仓内已有测试实际覆盖到多少。

### 4.1 对外行为矩阵

| 模式 | exception | kill | inactive-rank |
| --- | --- | --- | --- |
| noFT | scheduler 仍按原生错误路径处理 | 保持原生 fail-stop / Mooncake 隔离行为 | 由原生 active-rank path 处理 |
| FT_continue | 丢弃当前故障 batch、上报 fault，但 surviving ranks 继续跑 | 标记 dead，不全局 pause，继续服务 | 标记 dead，不全局 pause |
| FT_pause | 故障 rank 触发全局 pause 收敛，等待 `apply` | killed rank 记 dead，其余 rank 进入 paused | newly inactive rank 记 dead，其余 rank 进入 paused |

当前代码里的几个关键外部表现：

- `FT_continue + exception` 不是“静默忽略异常”，而是会丢弃当前 batch，并把 fault event 抛给控制面；只是控制面不触发全局 pause。
- `FT_pause + kill` 之后允许直接 `retry`，不会因为存在 dead rank 而拒绝恢复。
- `scale_down` 不只是标记 dead；它会真实更新 active-rank mask 和 DPC routing。
- 显式 `routed_dp_rank` 指向非 healthy rank 时，请求会被拒绝。

### 4.2 已有验证证据

当前仓内能直接看到的验证证据有三类：

1. 单元测试

- `test/srt/test_fault_tolerance_controller.py`

它覆盖了几个关键语义：

- `FT_pause + kill -> retry` 允许 dead rank 存在
- `scale_down` 的参数校验
- `continue + kill` 不进入 pause
- inactive mask 会影响 rank state
- 当前 `is_ft_supported_config()` 已经允许 Mooncake 多节点逻辑 FT

2. 已注册集成测试

- `test/registered/ep/test_mooncake_ep_small.py`

它验证的是：

- Pure DP 场景 kill 一个 scheduler 后，服务仍然可用
- Hybrid DP/TP 场景 kill 一个 scheduler 后，服务仍然可用
- 验收标准是 GSM8K 小规模评测仍高于阈值，而不是只看 HTTP 200

3. 手工测试入口

- `test/manual/ep/test_mooncake_expert_backup.py`

它说明 expert backup 这条链路已经进代码，但当前仍依赖手工 / 专项验证，而不是常规 CI 覆盖。

### 4.3 当前“已经实现”但“验证还不够”的部分

从代码存在性看，下面这些能力已经写进当前实现，但仓内自动化证据还不够强：

- `FT_continue + exception`
- `FT_pause + exception`
- inactive-rank 全链路
- `status/apply` API 契约
- live-rank `scale_down + shutdown` 分支
- `/fault_tolerance/*` admission gate
- `routed_dp_rank` 健康性约束

这部分不应该再写成“未实现”，更准确的说法是“已实现，但还缺系统性回归和发布级验证”。

## 5. 当前代码已经实现了哪些能力

如果对齐 `#8961` 及其拆分后的 M1~M6，当前代码已经具备的能力可以分成三层。

### 5.1 FT v5 控制面能力

- 显式 FT 开关和异常策略配置
- `healthy/paused/dead` rank 三态
- `status/apply` 外部管理接口
- `retry/scale_down` 恢复动作
- `exception/kill/inactive-rank` 三类故障输入
- `pause` 策略下的 admission gate
- `continue` 策略下的持续服务能力
- DPC 对 inactive / dead rank 的路由规避

### 5.2 Elastic EP 主路径能力

- Mooncake Backend / Mooncake EP
- active-rank mask 驱动的故障 rank 屏蔽
- fault-aware EPLB
- scheduler -> tokenizer -> DPC 的 DP-level fault awareness
- active-rank 变化在 scheduler / DPC / controller 三处联动

### 5.3 更靠近社区最终形态的弹性能力

需要先澄清一件事：[`#8961`](https://github.com/sgl-project/sglang/pull/8961) 本身已经在 2025-10-23 关闭，且 PR 描述里明确标注为 “superseded by `#15771`”。社区最终采用的不是“直接 merge 一个大 PR”，而是拆成以下 6 个里程碑分阶段合入主线：

| 里程碑 | PR | 状态 | 合入日期 |
| --- | --- | --- | --- |
| M1 Mooncake Backend / EP | [`#10423`](https://github.com/sgl-project/sglang/pull/10423) | merged | 2025-10-15 |
| M2 Elastic EP core + faulty-rank EPLB | [`#10606`](https://github.com/sgl-project/sglang/pull/10606) | merged | 2025-10-22 |
| M3 DP-level fault tolerance | [`#11657`](https://github.com/sgl-project/sglang/pull/11657) | merged | 2026-01-20 |
| M4 DRAM expert backup | [`#17374`](https://github.com/sgl-project/sglang/pull/17374) | merged | 2026-02-27 |
| M5 GPU P2P expert-weight exchange | [`#12068`](https://github.com/sgl-project/sglang/pull/12068) | merged | 2026-03-16 |
| M6 failed-rank rejoin | [`#15771`](https://github.com/sgl-project/sglang/pull/15771) | merged | 2026-04-28 |

本地代码仓中也已经能看到对应能力的代码痕迹：

- `enable_elastic_expert_backup`
- `ExpertBackupManager` / `ExpertBackupClient`
- `elastic_ep_rejoin`
- `model_runner.generate_weight_name_filter()` 驱动的 selective weight reload

所以如果只问“社区 roadmap 上的大块能力是否进代码了”，答案是：已经进了。

## 6. 当前代码明确没做，或者仍然受限的能力

当前代码里最清晰的限制来自 `is_ft_supported_config()`，也就是 FT 启动阶段的硬约束。

### 6.1 仍然明确不支持的组合

- `PP > 1`
- PD / disaggregation + FT
- NPU + FT
- `tokenizer_worker_num > 1`
- Ray engine + FT
- 非 Mooncake active-rank backend + FT

这些不是“效果可能不好”，而是当前代码明确拒绝的配置。

### 6.2 仍然没有对外承诺的语义

即使代码已实现 FT v5，也不能把下面这些能力写成已承诺：

- exactly-once 请求语义
- 自动请求重放
- 所有 in-flight kill 请求都必然成功
- 所有 fallback / shutdown 分支都已经被充分压测
- 所有拓扑下都能稳定重入恢复

更准确的描述是：当前实现的目标是“rank 级降级和恢复”，不是“请求级强一致恢复”。

## 7. 当前最重要的设计/实现偏差

这部分最值得工程师关注，因为它决定了文档应该相信代码还是相信旧设计稿。

### 7.1 设计稿说“false->true 不自动恢复”，当前代码已经会恢复

`controller.py` 当前的 `record_inactive_mask()` 在收到 `is_active=True` 且当前状态为 `DEAD` 时，会把 rank 直接恢复成 `HEALTHY`。

这和 v5 架构文档里“false->true 初版忽略，不自动恢复 dead rank”的表述不一致。

影响：

- 文档如果仍按旧说法写，会误导测试和运维。
- 这也说明当前代码已经部分吸收了 rejoin / 恢复后的状态回补语义。

### 7.2 设计稿说“初版只支持单机 / effective TP=1”，当前代码已经放宽

当前 `is_ft_supported_config()`：

- 不再检查 `effective TP == 1`
- 对 Mooncake backend 已允许逻辑多节点

同时，仓内 `test_mooncake_ep_small.py` 还覆盖了 Hybrid DP/TP 的 kill 后继续服务。

影响：

- 旧设计文档里的“effective TP>1 启动拒绝”已经不能代表当前代码。
- 面向开发工程师的总结必须以当前实现为准，而不是沿用 v5 初稿边界。

### 7.3 社区主线路径和本地 FT v5 接口形态不同

社区主线已经把 `#8961` 的里程碑能力分阶段合入，但主线公开接口仍主要围绕：

- Elastic EP / active ranks
- `pause_generation` / `continue_generation`
- rejoin / backup / EPLB

当前本地分支则额外引入了：

- `fault_tolerance/` 独立模块
- `/fault_tolerance/status`
- `/fault_tolerance/apply`
- `pause|continue` 明确外部策略语义

影响：

- 本地方案更完整，也更适合运维控制面接入。
- 但如果目标是最终“零摩擦合入社区”，这套 API 和状态语义需要额外 upstream 对齐，不是自动就能进主线。

## 8. 离最终合入社区的 GAP 还有哪些

如果把“最终合入社区”理解成“对齐 `#8961` 已经演进出的社区最终形态，而不是只在本地分支可用”，我认为 GAP 主要有四类。

### 8.1 接口形态 GAP

当前本地 FT v5 的最大差异不是底层机制，而是对外 contract：

- 社区主线合入的是 Elastic EP 路径
- 本地分支新增的是 FT v5 控制面 API 和显式状态机

这意味着后续必须二选一：

1. 把本地 `status/apply` contract upstream 成社区正式管理接口。
2. 或者把本地控制面能力压回社区现有的 Elastic EP / admin API 形态。

不解决这一点，代码再完整也只是“本地增强版”，不是社区标准形态。

### 8.2 文档一致性 GAP

当前最明显的问题是：

- 代码已经走到了多节点 / rejoin / active-mask 回补
- `v5_arch` 和 `v5_noft_impl` 仍保留一部分“初版限制”叙述
- 仓内公开文档还没有正式介绍 `--enable-fault-tolerance` 和 `/fault_tolerance/*`

这会直接造成开发、测试、SRE 三方对能力边界理解不一致。

### 8.3 自动化验证 GAP

当前仓内有：

- controller 单元测试
- kill-one-rank 级别的集成测试
- expert backup 手工测试

但缺的自动化回归非常关键：

- `FT_continue + exception`
- `FT_pause + exception`
- inactive-rank 真实注入
- `status/apply` API 契约
- `FT_pause + kill + retry/scale_down`
- live-rank shutdown path
- rejoin 与 FT 控制面交互
- 多节点和 RDMA 下的发布级套件

从工程合入角度看，这一项现在比“再补一个小能力”更优先。

### 8.4 支持矩阵 GAP

当前代码仍然明确排除了：

- PP
- PD/disaggregation
- NPU
- multi-tokenizer
- Ray
- 非 Mooncake backend

如果社区目标是“把 FT 写进主线并长期维护”，那这些边界至少要做到：

1. 文档里明确写清楚。
2. 启动时报错稳定。
3. 测试里把 unsupported matrix 固化下来。

否则很容易在后续回归中被误放开成半工作状态。

## 9. 建议如何对外表述当前项目状态

如果这份文档是给其他开发工程师看的，我建议对外统一成下面这套说法：

- 当前实现已经具备 rank 级 FT 控制面，不再只是 Elastic EP 内部机制。
- 当前实现已支持 `status/apply`、`pause/continue`、`retry/scale_down`、inactive-rank 感知和 DPC 路由收敛。
- 当前实现已经和 Mooncake Elastic EP、expert backup、rejoin 等社区主线能力并存。
- 当前最大风险不在“有没有代码”，而在“文档、接口和验证面是否已经收敛”。
- 如果目标是对齐社区 `#8961` 路线，剩余工作重点是 upstream 接口对齐、设计文档回写、自动化测试补齐和 unsupported matrix 固化。

## 10. 最后结论

当前这套 FT 框架已经完成了从“设计概念”到“控制面实现”的关键跨越。它的真实状态不是“还没做出来”，而是“已经做出来了，而且比最初 v5 设计稿更进一步”，只是还没有把实现、文档、测试和社区主线路径完全收拢成一个统一版本。

所以，面向工程团队最准确的结论是：

- 现在讨论的重点应该从“FT 要不要这么设计”切换到“当前实现要不要按这个接口继续推进，并如何补齐 merge-quality 的验证与文档”。
- 对齐 `#8961` 的核心能力面，代码层面已经大幅前进。
- 真正剩下的 GAP，是产品化和 upstream 化的 GAP，而不是从 0 到 1 的能力 GAP。
