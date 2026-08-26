# NPU + MC2 静态 Scale-down 实现说明

> 实现分支：SGLang `codex/review-0814-npu-ft-minimal`
>
> 核对提交：`127f5a6711f7133597d7ae496577427e0758d037`
>
> 公共分叉基线：`2abb1d2c37`
>
> 状态：核心代码已实现；尚未与当前异步 FT API 合并，尚缺 NPU 硬件 E2E

## 1. 准确的能力声明

该分支已经实现 NPU MC2 在 incident 后执行静态 `scale_down` 所需的主要设备和通信数据面改动，不应再描述成“纯设计/待开发”。

同时，它不是 GPU FT 能力的完整 NPU 等价实现。当前只声明：

- 策略：`pause`；
- 操作：静态 `scale_down`；
- 目标：让 survivor 重新建立 MLP-sync/EPLB 通信并以 compact EP topology 继续执行；
- 证据：代码和随分支提交的单元测试；
- 未闭环：最新 API/Manager 集成、NPU 硬件 E2E、Graph、retry、continue 和 rejoin。

“静态”指 active membership 在本次操作内从原集合单向减少；它不意味着未来不能重启服务恢复原拓扑，但当前分支没有在线 rejoin 契约。

## 2. 变更范围

相对 NPU FT 基线提交 `edaee40bf5`，MC2 适配主要涉及：

| 模块 | 作用 |
|---|---|
| `fault_tolerance/npu_communication.py` | FT TCPStore、compact Gloo/HCCL survivor group、collective timeout |
| `fault_tolerance/exceptions.py` | `NpuFTMLPSyncInterrupted` |
| `elastic_ep/npu_mc2.py` | MC2 elastic-info tensor 和物理 expert id 紧凑映射 |
| `elastic_ep/elastic_ep.py` | 初始化/原地更新 NPU MC2 elastic state |
| `eplb/process_group_context.py` | original rank 到 compact EPLB group rank 的上下文 |
| EPLB distribution/location/updater | survivor group collective、rank 映射和迁移 |
| `layers/moe/token_dispatcher/deepep.py` | 向 NPU dispatch/combine 传入 `elastic_info` |
| `model_executor/model_runner.py` | stop/restart、group rebuild、rebalance、elastic-info update |
| scheduler 与 DP-attention | MLP-sync survivor gather 和 FT pause 重入 |
| NPU utils | 保留 Ascend 内部格式的 staged copy |
| `server_args.py`、DPC | FT metadata store endpoint 和 support gate |

它采用窄范围 NPU 特化路径，没有引入旧文档设想的通用 `FtDeviceOps` 抽象，也没有修改所有设备后端。

## 3. 启动期准备

### 3.1 FT metadata store

DataParallelController 根据当前 task 的分布式 endpoint 建立独立 TCPStore。端口由 `port_base + 6` 派生，并通过 `fault_tolerance_metadata_port` 传给 scheduler。

DPC 侧 store 使用：

```text
is_master = true
wait_for_workers = false
```

各 scheduler 以 client 方式连接。每次 membership 变化使用独立 PrefixStore：

```text
npu-ft/<membership-bits>/<generation>/mlp-sync
npu-ft/<membership-bits>/<generation>/eplb
```

membership 和 generation 同时进入 prefix/group name，避免新旧 shrink 代际误加入同一 rendezvous。

### 3.2 初始通信对象

Scheduler 调用 `init_npu_ft_communication()` 保存：

- `original_rank`；
- `original_world_size`；
- 旧 MLP-sync group；
- collective timeout；
- 当前 active original ranks。

当前代码传入的是 `dp_rank/dp_size`，而 scale-down 的 `active_mask` 来自 scheduler global-rank 空间。若一个 DP 包含多个 scheduler rank，两者可能不等长。因此在新增明确映射或启动 gate 前，本版本把“每个 DP 一个 scheduler rank”视为隐含限制。

## 4. scale-down 设备事务

`ModelRunner.apply_fault_tolerance_scale_down(active_mask)` 的 NPU MC2 路径按以下顺序执行：

```text
torch_npu.npu.stop_device(gpu_id)
torch_npu.npu.restart_device(gpu_id)
NpuFTCommunication.rebuild(active_mask, npu_device)
copy active_mask into ElasticEP state
force EPLB rebalance
update MC2 elastic_info in place
snapshot active mask as last_active
```

### 4.1 为什么 stop/restart 在 survivor 上执行

scale-down 命令只发给 survivor。survivor 先停止并重启本地 NPU device，再建立新 HCCL group，避免继续使用 incident 前的设备/collective 状态。

这是一条重设备事务，应在真实 NPU 上验证耗时、stream/event 清理和失败模式。当前代码没有把它抽象为跨设备通用恢复动作。

### 4.2 compact survivor group

给定 original ranks `[0, 1, 2, 3]`，移除 1 后：

```text
active_original_ranks = (0, 2, 3)
original 0 -> compact 0
original 2 -> compact 1
original 3 -> compact 2
```

每个 survivor 通过 `active_ranks.index(original_rank)` 得到新 group rank，并创建：

- Gloo `mlp_sync_group`，world size 为 survivor 数；
- HCCL `eplb_group`，world size 为 survivor 数。

创建后先执行 Gloo barrier，再对 NPU tensor 执行一次 HCCL all-reduce warmup。只有全部 survivor 完成后才替换当前通信上下文。

该实现没有替换 SGLang 全局 `_TP` group；它只为易受 DP membership 影响的 MLP-sync 和 EPLB 建专用 survivor group，控制了改动范围。

## 5. MLP-sync 的固定槽位与有界超时

DP-attention 路径仍需要跨 DP 汇总 MLP 输出。NPU 适配用 survivor Gloo group 收集 compact 结果，再把结果写回按 original DP id 分配的固定槽位。

这样上层 tensor/layout 仍以 original DP 数为边界，通信 group 则只等待 survivor，不会等待 dead rank。

提交 `127f5a6711` 将 gather 改为异步 work，并执行有界等待：

```text
timeout = 5 seconds（当前 MLP-sync 路径硬编码值）
```

collective 抛异常或超时会转换成 `NpuFTMLPSyncInterrupted`。Scheduler 捕获后不继续使用不确定结果，而是重新进入 FT paused control loop。新 Gloo/HCCL group 自身的创建超时则使用 `fault_tolerance_timeout`，两者不是同一个参数。

这是防止旧组永久挂死的重要修复，但它只能证明超时控制路径存在；真实 HCCL/Gloo 故障行为仍需设备验证。

## 6. EPLB survivor context

`EPLBProcessGroupContext` 保存：

- 新 HCCL survivor group；
- `active_original_ranks`；
- original rank 与 compact group rank 的转换。

EPLB 的 all-reduce、broadcast 和 P2P 不再隐式使用 original world size，而是使用该上下文：

- collective world size 取 survivor 数；
- root/src/dst 先从 original id 映射成 compact id；
- inactive original endpoint 不参与迁移；
- expert location 仍能以 original rank 解释控制面和模型元数据。

NPU 内部格式 tensor 不能总用普通 contiguous copy。适配加入 staged copy helper，确保 expert migration/P2P 保留 Ascend format 语义。

## 7. MC2 `elastic_info`

### 7.1 固定布局

`NpuMC2ElasticInfo` 创建一个长度固定的 `int32` tensor：

```text
[header: 4]
[original_to_effective: original_ep_size]
[effective_to_original: original_ep_size]
```

header 当前含义：

```text
0: has_shrunk
1: effective_ep_size
2: reserved (0)
3: effective_ep_size * num_local_physical_experts
```

inactive original rank 在 `original_to_effective` 中为 `-1`；`effective_to_original` 未使用的尾部也为 `-1`。

### 7.2 原地更新

membership 变化时调用 `tensor.copy_(new_payload)`，不会替换 tensor 对象。单元测试显式检查 `data_ptr()` 在更新前后保持不变。

固定地址是 Graph 兼容的必要条件之一，因为 dispatch/combine 捕获时可以引用同一参数地址；但它不是 NPU Graph 恢复已通过的充分证据。尚需验证 Graph 是否读取更新后的内容、HCCL communicator 重建能否与已捕获图共存，以及 replay 是否触达 stale 资源。

### 7.3 物理 expert id 紧凑化

原始 physical expert id 按：

```text
original_rank = physical_id // num_local_physical_experts
local_id      = physical_id %  num_local_physical_experts
```

映射为：

```text
compact_id = original_to_effective[original_rank]
             * num_local_physical_experts
             + local_id
```

原 rank 已失效或原 id 本身无效时返回 `-1`。DeepEP NPU dispatch/combine 同时收到 `elastic_info`，从而在 compact effective EP 空间解释 expert placement。

## 8. 已实现的单元测试意图

分支包含以下专用测试：

| 文件 | 覆盖意图 |
|---|---|
| `test_npu_mc2.py` | sparse survivor 映射、physical id compact、elastic tensor 地址稳定 |
| `test_process_group_context.py` | original/compact rank 映射和 inactive endpoint |
| `test_npu_mlp_sync_timeout.py` | gather 异常/超时转成 FT interruption |
| `test_controller.py` 增量 | NPU + MC2 support gate |
| `test_scheduler_control.py` 增量 | scheduler 捕获 MLP interruption 并进入 FT loop |

本次文档核对环境没有安装 `torch`，无法执行这些测试；它们只能列为“分支内存在的测试代码”，不能写成“本机已通过”。硬件证据缺口见 [VALIDATION.md](VALIDATION.md)。

## 9. 与 GPU 当前 API 的差异

NPU 分支仍在旧控制面上，当前行为包括：

| 项目 | NPU 分支现状 | v0.1.0 目标 |
|---|---|---|
| apply 完成模型 | 同步等待，HTTP 200/400 | 异步 202 + status poll |
| scale 参数 | `params.ranks` | `params.removed_dp_ranks` |
| status | `ranks[].rank/state` | `schema_version/total_engines/engines[].id/status` |
| 操作 request id | 无当前聚合语义 | success/failure 都记录 `last_ft_request_id` |
| inactive route | 旧异常/400 路径 | HTTP 503 admission |
| 状态内部模型 | 旧 Manager/controller | expected/alive/native/pending 分离 |
| rejoin | 未验证旧继承路径 | 当前 GPU 自动 rejoin 契约 |

在 port 完成前，不能用最新 GPU API 用例直接声明 NPU 兼容。

## 10. 明确不支持或未验证的能力

### 10.1 `retry`

NPU 分支的 retry 没有调用 `stop_device/restart_device`、`NpuFTCommunication.rebuild()`、EPLB survivor context 或 MC2 elastic-info 更新。因此 retry 不是当前 NPU 契约。

### 10.2 `continue`

代码没有完整证明 survivor 可在 incident 处置前安全跨过旧 MLP-sync/HCCL collective。没有显式 gate 不代表支持；v0.1.0 不声明 NPU continue。

### 10.3 rejoin

compact group generation、device restart、MC2 mapping 和重新加入 original rank 的逆向事务没有实现并验证。没有 `recover` API，也不能沿用 GPU rejoin 结论。

### 10.4 请求 retract/replay

旧设计曾提出获取 in-flight request、retract、重建 KV 状态再 replay。当前分支没有这些实现。受故障请求使用既有 discard/failure 语义。

### 10.5 NPU Graph

目前只有 elastic-info tensor 地址稳定的测试意图，没有 NPU 硬件 Graph capture/replay artifact，不能写成“无需重捕图已验证”。

### 10.6 多 scheduler-rank DP

当前代码把 `dp_rank/dp_size` 与 global active mask 直接组合，缺少 whole-DP 展开/压缩适配。除非用户确认目标拓扑并补充代码 gate，本版本仅把每 DP 单 scheduler-rank 视为当前可解释配置。

## 11. 集成到当前控制面的工作清单

1. 把 NPU 数据面改动移植到 `codex/ft-vllm-api-refactor` 的 Manager/controller 基线，而不是反向恢复旧同步 API。
2. 让 scale-down 统一接收 whole-DP `removed_dp_ranks`，由控制面生成 NPU 所需 survivor mask。
3. 明确 NPU 通信对象使用 DP mask 还是 global-rank mask，并在启动时验证尺寸和 rank 空间。
4. 接入异步 202、唯一 request id、status 聚合错误和 operation guard。
5. 统一 inactive route 503 与外部 LB 更新契约。
6. 保留 NPU scale-down 专属 stop/restart、compact group 和 elastic-info 事务。
7. 对底层异常沿用 fail-stop，不把部分 group 创建当成可恢复成功。
8. 在 NPU 硬件完成冷启动、故障注入、缩容、精度、持续请求、Graph、collective timeout 和重复冷启动。

## 12. 发布判定

只有同时满足以下条件，才能把能力矩阵从“代码已实现”升级为“集成并验证”：

- NPU 分支对齐当前 API 和三态模型；
- 配置 gate 与实际 rank 空间一致；
- 非 rank0 incident + 静态 scale-down 在 NPU 硬件成功；
- survivor 推理、显式路由、精度、EPLB rebalance 和持续请求通过；
- inactive target 的 HTTP/status 语义与 GPU 契约一致；
- Graph 若被声明支持，必须有 capture + shrink + replay artifact；
- 所有日志、命令、提交、配置和输出 artifact 可回取。
