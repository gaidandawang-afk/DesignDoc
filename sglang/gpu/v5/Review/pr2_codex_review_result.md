# PR #2 Review 处理结果

## 范围与证据

- 当前分支：`ft_preview`，HEAD `47660ff65`，相对 `origin/ft_preview` ahead 1。
- Review 对象：`da38a58ba`；当前分支包含其后的 `531612e05` 和 `47660ff65`，不能按旧 review head 机械回退。
- 设计依据：`Spec/sglang_fault_tolerance_design_v5_arch_zh.md`，重点是 6.2 kill、7.1 noFT parity、7.2 command fanout、7.3 routing ack、7.4 active-mask 单一入口和 7.5 topology-change park。
- 历史提交：`325bc1f15`（noFT Mooncake rank loss）、`8baad7fdf`（删除 post-scale_down park）、`bebb24d81`（仅 topology change park）、`b8c798969`（fallback CPU broadcast）、`e8eeaf7dd`（routing ack/watchdog）、`531612e05`（保留 live expert replicas）。
- 历史问题：[#22](https://github.com/gaidandawang-afk/stackoverflow/issues/22)、[#31](https://github.com/gaidandawang-afk/stackoverflow/issues/31)、[#33](https://github.com/gaidandawang-afk/stackoverflow/issues/33)。

## 已处理

1. **Review 3、6：ElasticEP 参数和 mask 转换已简化。** `_refresh_ep_members()` 不再接收 `allow_missing_buffer_in_fallback`，而是直接读取 `MOONCAKE_EP_FORCE_FALLBACK`；`apply_active_rank_mask()` 直接把 `List[bool]` 转为目标 dtype tensor。功能影响是 fallback 模式下缺少尚未初始化的 EPBuffer 时统一跳过刷新，正常模式仍抛错；bool 到 int32 仍得到 0/1，不改变 mask 值。
2. **Review 7：engine watchdog 的进程身份已明确。** `dp_size == 1` 时 engine 直接监控 scheduler，`dp_size > 1` 时只监控 DPC；进程名据此区分，FT callback 只接管 `scheduler_*`。这样 DPC/detokenizer 崩溃仍走原 fail-stop，不会被 FT callback 误吞。
3. **Review 9、11：watchdog 退出去重改为 `index -> pid`。** 同一进程只上报一次；如果同一槽位恢复为存活的新 pid，记录会被移除。监控循环重新每轮读取 process list，不再使用初始化时冻结的 sentinel/remaining 集合。
4. **Review 14、24：FT unavailable JSON 已统一。** middleware、status 和 apply 都使用 `fault_tolerance_unavailable_response()`，其 body 统一由 `ft_failure()` 构造。HTTP 503 和 `fault_tolerance_disabled/paused` 语义不变。
5. **Review 21：fallback metadata broadcast 已简化。** CPU process group 只获取一次并复用。函数本身保留，因为 `b8c798969` 解决的是故障恢复时 device-group broadcast 不可靠的问题，不是 precision debug。
6. **Review 23：不合理的 `nnodes == 1` FT gate 已由当前 HEAD `47660ff65` 删除。** rejoin/多节点不应被这一层无条件拒绝；其他不支持组合仍由 PP、backend、PD、NPU、multi-tokenizer、Ray 校验拦截。
7. **Review 30、40、43：无功能价值的占行内容已收敛。** 删除 DPC shutdown 和 scheduler FT-command 的逐次日志；`routed_dp_rank` 校验压为单行。只减少日志噪声，不改变控制流。
8. **Review 44：`SGLANG_FT_DISABLE_PARK_IDLE` 已删除。** 当前 park 只在 active mask 实际变化前发生，并在同一恢复事务或首个请求前 resume；历史 #22 反对的是额外 post-scale_down park，不是 topology-change barrier。

## 已删除

1. **Review 2、15、16、18-20、25、50、57：precision debug 全部删除。** 包括 `SGLANG_FT_PRECISION_DEBUG`、`FTPrecisionDebug`、tensor/hash/fingerprint summary 及其日志。依据是这些内容来自 `e2511905e` 的临时定位提交；问题 #31/#33 已完成根因定位，正式路径无需保留。
2. **Review 26：Mooncake dispatch/combine debug 日志删除。** `_last_active_ranks_signature` 不删除，原因见“保留”部分。
3. **Review 52、55：ModelRunner forward exception 注入删除。** 删除环境变量入口、跨 rank fault flag、done-file 写入和主动抛错；正式推理路径不再包含测试故障注入。
4. **Review 58：本 PR 新增测试删除。** 删除 `test/srt/test_fault_tolerance_controller.py`，并移除 EPLB inactive-rank 三个新增用例；在实现旁保留后续回归覆盖 TODO。现有 baseline 的 `maybe_inject_test_rank_fault()` 不属于本 PR diff，因此未越界删除。

## 已回退

1. **Review 1：`ElasticEPState.reset()` 回退为原地 `fill_(1)`。** 新建 tensor 虽然值相同，但会替换对象引用；原地更新保持其他组件持有的 tensor 引用有效，因此原实现更安全。
2. **Review 8、10、12：`SubprocessWatchdog` 恢复原轮询和 `_check_processes()` 语义。** FT 只增加可选 `on_exit` 拦截点；未被 callback 处理的非零退出仍发送 SIGQUIT。模块注释保留原说明，仅追加 callback 特性。
3. **Review 51：memory imbalance warning 恢复原 `else` 结构。** 开关开启时抛错，关闭时 warning；没有理由改写已有代码形态。
4. **上一轮误处理回退：保留 lifespan 中的 `auto_create_handle_loop()`。** scheduler 可能在首个 HTTP 请求前退出，而 watchdog 线程只能通过已存在的 tokenizer event loop 安排 FT task；删除预创建会让该场景退回 fail-stop。
5. **上一轮误处理回退：保留 DPC `_ensure_active_rank_available()` 和 `PendingFTCommand` 所在位置。** 前者避免全 inactive 时仍更新负载预算；后者持有 `asyncio.Future`，属于 tokenizer 的 ack 编排状态。只搬数据类会把精确类型退化为 `Any`，但不会真正减少 tokenizer 控制逻辑。

## 保留但说明理由

1. **Review 4、5：保留 native-DP 一元素 mask 折叠和后续长度校验。** Spec 7.4 明确：每个 native-DP scheduler 的 ElasticEP state 可能只有一元素，而控制面携带全局 DP mask，本地应折叠为 `[True]`；第二个校验仍负责捕获 `numel() > 1` 的真实长度错误，并非恒为 false。
2. **Review 13：保留 lifespan 事件循环预创建。** `handle_ft_process_exit()` 在 `event_loop is None` 时必须返回 false，watchdog 随后执行原 SIGQUIT；预创建是首请求前 kill 能进入 FT 状态机的必要条件，且只在 FT enabled 时执行。
3. **Review 17、22、26、53、54、56：保留 `531612e05` 的 live-replica 路径。** 问题 #33 已验证：有冗余 live replica 时强制 EPLB 会改变 BF16 grouped-GEMM slot/order并造成 token 漂移；静态 12/12、动态 21/21（其中 post-kill 14/14）通过。当前逻辑先过滤 inactive candidate，仅在缺失 live replica 时 fallback EPLB；`divmod`/remainder 校验防止错误 owner-rank 计算；Mooncake signature 变化时重置 first-dispatch，避免沿用旧 topology 状态。
4. **Review 27-33：保留 DPC 到 tokenizer 的反向链路和 DPC scheduler watchdog。** `dp_size > 1` 时 engine 只拥有 DPC 进程，scheduler 是 DPC 子进程，因此 kill 只能由 DPC 上报；DPC 还必须返回本地 shutdown ack 和 routing-update ack。控制面状态机仍在 Tokenizer/FT manager，DPC 只负责进程所有权、路由和本地动作，两者职责不冲突。
5. **Review 29：保留 `send_to_tokenizer is not None` guard。** 非零 node 明确把该 socket 设为 `None`；guard 防止多节点或非主控制进程误发 ack/event。
6. **Review 31：保留 DPC-local shutdown 顺序。** scale_down 必须先让 live schedulers 应用 mask、resume survivors、提交 routing inactive，再终止目标进程并 ack；提前 terminate 会让目标 rank 无法参与 mask fanout，晚于响应则产生请求竞态。
7. **Review 34-37：保留 liveness 刷新、active-rank guard 和 inactive-aware routing。** watchdog 按周期运行，dispatch 前刷新可关闭轮询间隔内仍把请求发给已退出进程的窗口；全 status=false 是 Spec TC-v5-019 的明确场景，必须 reject，不能 round-robin 空转；TOTAL_REQUESTS/TOKENS 必须在预算 dispatch 前完成该检查。
8. **Review 38：保留 exception batch discard。** Spec 6.6 要求 exception 请求释放 KV、返回 503 且不重放；只 pause 而不清理会遗留 batch/KV 状态。正常 noFT event loop 不进入该包装。
9. **Review 39：保留 `_ft_rank()`。** native DP 的 `dp_rank` 可能为 `None`，FT status/command 使用 logical rank 0；该归一化同时用于 exception event 和 command target matching，避免两处重复并保持一致。
10. **Review 41：保留 topology-change park 和 response barrier cleanup。** `bebb24d81` 把 park 限定在 active mask 变化前，用于排空 overlap result queue 和空 batch；`8baad7fdf` 与问题 #22 已删除会破坏 Mooncake collective 顺序的额外 post-scale_down park。
11. **Review 42、47、49：保留 `PendingFTCommand`、`_ft_create_task()` 和 callable/coroutine wrapper。** pending 对象聚合每个 target rank 的 ack/failure；task 被 `asyncio_tasks` 持有并通过 done callback 移除；wrapper 同时服务传入 bound callable 的 `handle_loop` 和已创建 coroutine 的 FT task。
12. **Review 45、46、48：本轮不做 tokenizer 大迁移。** `_ft*` 方法直接依赖 tokenizer event loop、ZMQ socket、request admission 和 pending futures；完整迁移需要定义独立 coordinator 的生命周期与 IPC 边界，属于单独重构。只移动少数函数或数据类会形成跨对象回调，不能真实降低侵入。

## 仍需人工确认

1. **Review 45、46、48 的结构性重构。** 如果社区要求 TokenizerManager 明显减行，应单独设计 `FaultToleranceCoordinator`，而不是在本 PR 的 review cleanup 中拆一半；需要 maintainer 先确认所有权边界。
2. **Linux/GPU 功能回归。** 必须复跑 noFT kill parity、FT_continue kill、FT_pause retry/scale_down、连续 scale_down、首请求前 static kill、DP-attention 和 rejoin。当前 Windows 环境不能验证 Mooncake/NCCL 语义。
3. **测试删除后的覆盖策略。** Review 要求不保留本 PR 新增测试，但 active-rank filtering、watchdog callback 和 controller 状态机仍需要由外部验证脚本或后续可接收的测试 PR 覆盖。
4. **格式工具。** 本机没有 `ruff`，需要 CI 执行仓库正式 lint/format 配置。

## 自检结果

1. `SGLANG_FT_PRECISION_DEBUG|FTPrecisionDebug|_debug_tensor_summary|_logical_count_debug_summary`：`NO_MATCH`。
2. `allow_missing_buffer_in_fallback`：`NO_MATCH`。
3. forward exception 注入和 `SGLANG_FT_DISABLE_PARK_IDLE`：`NO_MATCH`。
4. `apply_active_rank_mask`：保留 1 个 helper；scheduler 有 FT command 和 noFT ActiveRanksOutput 两个调用入口。
5. `python -m compileall -q python/sglang/srt`：通过。
6. controller 隔离加载状态机 smoke：通过，包含 kill -> pause -> retry 和 `nnodes=4` 配置校验。
7. watchdog/EPLB pytest：Windows 收集阶段因 `ModuleNotFoundError: resource` 阻断，未执行测试体。
8. `git diff --check`：通过。
9. `ruff`：本机未安装，未运行。
