# SGLang DP-only Fault Tolerance v0.1.0

> 更新时间：2026-08-26
>
> 文档状态：实现对齐基线，不是已发布版本或单一合并分支的声明

## 1. 这组文档解决什么问题

本目录统一描述 SGLang、Mooncake GPU 弹性 EP 和 NPU MC2 适配三条开发线当前已经实现、已经验证和仍待集成的 DP-only 容错能力。

旧版 `gpu/v7` 与 `npu` 文档确实已经部分过时：GPU 文档仍使用 `disabled`、显式 `recover` 和同步 FT API；NPU 文档主要停留在方案阶段，未反映 `codex/review-0814-npu-ft-minimal` 已完成的静态缩容代码，也包含当前实现没有采用的请求回放、统一设备抽象和“固定通信组只传 mask”等设想。本版本以代码和测试证据为准重新编排，不把设计意图写成已验证事实。

v0.1.0 是一个跨分支的文档版本：

- GPU 控制面和 API 以 SGLang `codex/ft-vllm-api-refactor` 为准；
- GPU 数据面恢复以 Mooncake `codex/mooncake-nohca-ft` 为准；
- NPU 静态缩容以 SGLang `codex/review-0814-npu-ft-minimal` 为准；
- 三者尚未全部合并到同一个源码提交，尤其 NPU 分支仍使用旧版同步 FT API。

因此，“v0.1.0 支持某能力”必须同时注明设备、分支和证据等级。

## 2. 事实来源

| 组件 | 分支 | 本次核对提交 | 用途 |
|---|---|---:|---|
| SGLang GPU FT/API | `codex/ft-vllm-api-refactor` | `8a8860da82cf1ef4191c865f97715d2779c61743` | 当前三态模型、异步 API、路由拦截、自动 rejoin |
| Mooncake GPU FT | `codex/mooncake-nohca-ft` | `d727290c86e4c821a6fc4c22848ae5c0f269f4f5` | no-HCA 恢复、CUDA Graph 延迟建连修复 |
| SGLang NPU MC2 FT | `codex/review-0814-npu-ft-minimal` | `127f5a6711f7133597d7ae496577427e0758d037` | NPU 静态 scale-down、存活组重建、MC2 elastic info |
| server_tool 契约 | `codex/test/ft-vllm-api-refactor` | `7367ae315f524477abca46c51c9b082658d58fa1` | GPU 场景、断言和 artifact 索引 |

提交号用于固定本文审阅时的事实，不应被写入运行 profile 或测试契约作为长期配置。执行时仍应从所选分支的 HEAD 获取待测提交。

## 3. 当前结论

### 3.1 GPU + Mooncake

GPU 路线已经形成可验证闭环：故障检测、中心化 pause/continue、`retry`、`scale_down`、外部请求路由拦截、存活 DP 恢复、自动 rejoin，以及 CUDA Graph 下 Mooncake 延迟建连恢复均已有对应代码和测试证据。

当前公开状态只有 `healthy`、`unhealthy`、`dead`。不存在 `disabled` 状态，也不存在显式 `recover` 指令。进程重新出现且 Mooncake native active mask 恢复后，由 Manager 自动重新开放该 DP。

FT 写接口是异步接受模型：合法请求返回 HTTP 202 和 `request_id`，最终结果通过 `/fault_tolerance/status` 中同一 `last_ft_request_id` 及目标拓扑共同确认。仅观察拓扑变化不能证明本次请求成功。

被 scale-down 的 DP 不再属于路由集合。显式指定该 DP 的推理请求由 SGLang admission 层拒绝，当前契约是 HTTP 503 和 `routed_dp_rank=<N> is not active`。这属于 SGLang 的接口语义，不应错误归因于 vLLM PR #46370；外部 LB 仍必须停止向被移除 endpoint 路由。

### 3.2 NPU + MC2

NPU 分支不再只是设计稿。它已经实现静态 scale-down 的关键数据面事务：

1. NPU device stop/restart；
2. 为存活 rank 建立新的紧凑 Gloo MLP-sync 组和 HCCL EPLB 组；
3. 将 EPLB 的 original rank 映射到 compact group rank；
4. 重排物理 expert id 到存活 EP 命名空间；
5. 原地更新固定地址的 MC2 `elastic_info` tensor；
6. 强制 EPLB rebalance，并为 MLP-sync collective 增加有界超时和重新进入 FT pause 的路径。

但该分支从较早公共基线分叉，仍使用同步 HTTP 200/400 API、`params.ranks` 和旧 status schema。它没有实现或验证 NPU `retry`、`continue`、rejoin、透明请求回放及 NPU 硬件 CUDA/ACL Graph 端到端。因此 v0.1.0 对 NPU 的准确说法是“静态缩容代码已实现，尚待与当前控制面集成并完成硬件 E2E”，而不是“与 GPU 功能等价”。

### 3.3 集成状态矩阵

| 能力 | GPU + Mooncake | NPU + MC2 | 备注 |
|---|---|---|---|
| 异常后中心化 pause | 已实现并验证 | 继承旧基线；仅静态缩容链路使用 | NPU 尚未对齐最新 Manager/API |
| `retry` | 已实现并验证 | 不支持 | NPU retry 未执行 device/group/MC2 重建 |
| 静态 `scale_down` | 已实现并验证 | 已实现代码 | NPU 缺硬件 E2E 和新 API 集成 |
| scale-down 后路由拦截 | HTTP 503 已验证 | 旧分支为旧语义 | 目标契约应统一为 GPU 当前语义 |
| 自动 rejoin | 已实现并验证 | 未实现/未验证 | 无显式 `recover` |
| `continue` 丢弃受影响请求后恢复 | 已实现并验证 | 未声明支持 | 不应由“代码未显式禁止”推断支持 |
| 整节点 watchdog | 逻辑节点租约已验证 | 未验证 | GPU E2E 使用同一物理机上的逻辑节点，不等价于真实下电 |
| Graph 恢复 | CUDA Graph 已验证 | 仅固定 tensor 地址单测意图 | NPU 无硬件 graph 证据 |
| 请求 retract/replay | 未采用；使用 fail/discard 语义 | 未实现 | 旧 NPU 方案中的设想已删除 |

## 4. 文档导航

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 统一状态机、API、控制流、设备边界和规范性约束 |
| [GPU_MOONCAKE.md](GPU_MOONCAKE.md) | GPU/Mooncake 实现、watchdog、rejoin、CUDA Graph 与依赖配对 |
| [NPU_MC2_STATIC_SCALE_DOWN.md](NPU_MC2_STATIC_SCALE_DOWN.md) | NPU 已实现静态缩容的代码路径、限制和移植事项 |
| [NPU_MC2_DESIGN_RATIONALE.md](NPU_MC2_DESIGN_RATIONALE.md) | GPU 与 NPU 路线为何不同，以及旧方案取舍的更新 |
| [NPU_MC2_COMPACTED_TOPOLOGY.md](NPU_MC2_COMPACTED_TOPOLOGY.md) | original/compact rank、通信组和 expert 映射成本 |
| [VALIDATION.md](VALIDATION.md) | 精确提交、GPU 用例矩阵、NPU 证据缺口和发布门槛 |

## 5. 旧文档迁移映射

| 旧路径 | 新路径 | 处理方式 |
|---|---|---|
| `sglang/SGLang_fault_tolerance.md` | `sglang/v0.1.0/README.md` | 改为跨分支版本总览 |
| `sglang/gpu/v7/SELF_PAUSE_WHOLE_DP_FT.md` | `sglang/v0.1.0/ARCHITECTURE.md` | 删除旧状态/API，重写统一契约 |
| `sglang/gpu/v7/SELF_PAUSE_WHOLE_DP_FT_DESIGN.md` | `sglang/v0.1.0/GPU_MOONCAKE.md` | 以当前实现和证据重写 |
| `sglang/npu/SGLang-NPU-DP-FT-GPU兼容适配方案.md` | `sglang/v0.1.0/NPU_MC2_STATIC_SCALE_DOWN.md` | 从预研方案改为实现说明 |
| `sglang/npu/SGLang-NPU-MC2-容错适配方案.md` | `sglang/v0.1.0/NPU_MC2_DESIGN_RATIONALE.md` | 保留有效决策，删除未落地设想 |
| `sglang/npu/sglang_mc2_dp_attention_compacted_cost_zh.md` | `sglang/v0.1.0/NPU_MC2_COMPACTED_TOPOLOGY.md` | 对齐实际 compact group 实现 |

Git 历史保留原文及其演进；本目录不复制一份可能继续漂移的旧版。

## 6. v0.1.0 的规范性边界

以下约束适用于统一后的目标控制面：

- 容错粒度是整个 DP replica，不允许只移除 DP 内单个 scheduler rank；
- 对外状态固定为 `healthy`、`unhealthy`、`dead`；
- 写操作固定为 `retry` 和 `scale_down`，不提供 `recover`；
- 同一时间最多一个 FT 操作，Manager 是唯一控制面提交者；
- `scale_down` 只向存活 rank 下发数据面调整，被移除 rank 保持 `DEAD`；
- 路由集合必须同时满足 expected、process alive 和 backend native active；
- 外部 LB 必须和 SGLang 路由状态同步，SGLang 的 503 是最后一道保护而不是 LB 替代品；
- 不能把 fail-stop、健康请求成功或拓扑变化误报为 FT 指令成功；
- GPU 已验证结论不能自动外推到 NPU；NPU 代码存在也不能替代硬件 E2E。

NPU 当前实现对一个 DP 内含多个 scheduler rank 的支持边界尚未闭合：通信组初始化使用 `dp_rank/dp_size`，缩容 mask 则来自 global scheduler-rank 空间。v0.1.0 暂将“每个 DP 一个 scheduler rank”列为代码隐含约束，直到实现增加明确 gate 或完成多 rank DP 验证。

## 7. 下一阶段工作

1. 将 NPU MC2 适配 rebase/port 到当前 GPU FT/API 分支，解决 Manager、status schema、异步 `request_id` 和 503 admission 语义差异。
2. 为 NPU 配置增加与真实拓扑假设一致的 fail-fast gate，或实现 whole-DP global-rank mask 到 compact DP group 的完整映射。
3. 在 NPU 硬件上执行冷启动、注入非 rank0 故障、静态缩容、精度、持续请求、Graph 和失败超时用例，保存 artifact。
4. 在真实多节点 GPU 环境补一次物理节点下电/网络隔离，区分逻辑租约 E2E 与真实主机故障证据。
5. 只在上述集成和设备验证完成后，升级 NPU 能力矩阵；不得提前把目标设计写成已支持能力。
