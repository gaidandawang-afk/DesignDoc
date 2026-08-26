# NPU MC2 容错设计依据与路线取舍

> 本文解释 v0.1.0 为什么没有要求 NPU 完全复制 GPU/Mooncake 的内部实现。规范性接口以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准，NPU 已实现代码以 [NPU_MC2_STATIC_SCALE_DOWN.md](NPU_MC2_STATIC_SCALE_DOWN.md) 为准。

## 1. 结论先行

GPU 与 NPU 应共享控制面语义，不必共享通信实现：

- 对外都使用稳定 original DP id、三态 status、异步 request id 和 whole-DP 操作；
- GPU/Mooncake 在稳定 rank namespace 内用 active mask 屏蔽失效 peer；
- NPU/MC2 当前为 MLP-sync 和 EPLB 建 compact survivor group，同时向 MC2 kernel 传固定地址的 `elastic_info`；
- NPU 分支没有整体替换全局 `_TP` group，也没有实现透明请求 replay；
- “不重捕图”目前只是通过固定参数地址降低重捕图需求的设计方向，尚无 NPU 硬件证据。

旧文档把若干预研选项写成了接近既定实现的表述。本版将其重新分为“已经采用”“明确未采用”和“仍需验证”。

## 2. 共同的控制面，不同的数据面

### 2.1 共同部分

设备无关的责任包括：

- incident 识别和 whole-DP 扩展；
- expected/process/native 三类事实分离；
- pause/continue 策略选择；
- 单事务 operation guard；
- `retry`/`scale_down` API；
- status、request id、路由和 LB 契约；
- fail-stop 与 artifact 证据要求。

这些部分不应因为设备差异而分裂成两套外部 API。

### 2.2 设备特化部分

设备事务必须尊重后端真实能力：

| 问题 | GPU/Mooncake | NPU/MC2 当前实现 |
|---|---|---|
| 失效 peer 屏蔽 | 稳定 rank + active mask | compact survivor groups + elastic info |
| 本地设备恢复 | Mooncake connection/Buffer 恢复 | `torch_npu.npu.stop_device/restart_device` |
| MLP-sync | Mooncake/既有路径 | 新建 survivor Gloo group |
| EPLB collective | active mask/后端弹性状态 | 新建 survivor HCCL group + rank context |
| kernel topology | active ranks | fixed-size `elastic_info` tensor |
| Graph 证据 | CUDA Graph E2E 已有 | 仅 tensor 地址稳定测试意图 |

强行把两者抽象成完全相同的 hook，会隐藏最关键的设备顺序、communicator 语义和失败边界。

## 3. 评估过的两类路线

### 3.1 路线 A：固定 original group，仅传 active mask

思路是保持所有 communicator 的 world size 和 rank 不变，collective/kernel 根据 mask 忽略 dead peer。

优点：

- original id 与 group rank 完全一致；
- 外部状态、日志和模型元数据不需要转换；
- 如果后端原生保证 masked collective 不等待 inactive peer，Graph 和 communicator 变化较少。

风险：

- 需要 MC2/HCCL 对 inactive peer 有可证明的非阻塞语义；
- 旧 collective 可能在 device/peer 故障后已经进入不可恢复状态；
- EPLB broadcast/P2P 仍可能隐式等待原 world；
- 单纯传 mask 不能修复 Python 层使用固定 group 的同步点。

GPU Mooncake 的 native elastic 语义更接近这条路线；不能据此推断 NPU MC2/HCCL 同样成立。

### 3.2 路线 B：为受影响 collective 建 compact survivor group

思路是保留 original id 作为控制面命名，在真正需要跨 DP collective 的位置创建只含 survivor 的 group。

优点：

- collective world 中不含 dead peer；
- barrier/all-reduce/gather 的等待集合明确；
- 可以用 generation 隔离多次 shrink；
- original 与 compact 映射可集中在 EPLB context 和 elastic info。

成本：

- 引入两套 rank 空间；
- root/src/dst 必须显式映射；
- expert id 和物理容量要重算；
- communicator 创建是重操作；
- rejoin 需要逆向 membership 事务，复杂度高。

NPU 分支选择的是受控范围的路线 B：只重建 MLP-sync Gloo 与 EPLB HCCL 专用组，不整体重建 SGLang `_TP`。

## 4. 为什么还需要 MC2 `elastic_info`

compact Python process group 解决“哪些 rank 参加 collective”，但 MC2 dispatch/combine 还需要知道：

- 当前 effective EP world size；
- original rank 到 effective rank 的映射；
- effective rank 到 original rank 的逆映射；
- survivor 总物理 expert 数；
- 某个 physical expert id 是否已失效。

因此分支采用固定大小的 `elastic_info` tensor。它把 original topology 的最大尺寸保留在布局中，只原地修改 payload。

这是两种诉求的折中：

- 控制面和模型持久元数据仍使用 original id；
- MC2 kernel 使用连续 effective id；
- Graph 可继续引用同一 tensor 地址；
- 未使用槽位以 `-1` 明确标记，而不是改变 tensor shape。

## 5. 不整体替换 `_TP` 的原因

旧 compacted-rank 分析假设需要重建全局 `_TP` group，影响 attention、model-parallel state、所有 collective 和缓存。实际实现避免了这一步。

当前 NPU 分支只对已知会跨 DP 等待 dead peer的路径做定点替换：

- DP-attention 的 MLP-sync 使用 survivor Gloo group；
- EPLB 使用 survivor HCCL group；
- MC2 kernel 使用 elastic-info 映射。

上层固定槽位、original DP id 和其他并行组保持不变。这降低了改动面，但也意味着必须证明没有遗漏仍使用旧 world 的跨 DP collective。硬件 E2E 的价值正在这里：源码审阅无法证明运行时所有隐式 collective 都已覆盖。

## 6. DP-attention 数据依赖

DP-attention 并不意味着所有计算都天然 DP-local。常见路径是 attention 在每个 DP 内处理本地 token，而 MLP/EP 需要跨 active DP 协调 token/expert 结果。

旧文档提出通过 DP-local LM head、禁用跨 DP logits gather 来减小故障面，这仍是有价值的配置原则，但不能把它写成 NPU 分支已经彻底消除所有跨 DP 通信。当前代码明确保留并改造了 MLP-sync，这本身证明仍有跨 DP 依赖。

正确表述是：

- 能做 DP-local 的输出路径应保持 local；
- 必要的 MLP/EPLB collective 必须切换到 survivor group；
- 固定 original DP tensor 槽位与 compact communication rank 之间必须显式映射；
- 是否还有其他跨 DP collective 要靠 instrumentation 和硬件测试确认。

## 7. Graph：必要条件与充分条件

### 7.1 已实现的必要条件

`NpuMC2ElasticInfo.update()` 使用 `copy_` 原地更新，保持 tensor `data_ptr()`。这避免了因为 topology 变化自动替换 kernel 参数地址。

### 7.2 尚未证明的条件

以下问题没有 NPU 硬件 artifact：

- Graph replay 是否每次读取更新后的 elastic-info 内容；
- stop/restart device 后旧 Graph executable 是否仍有效；
- 新 HCCL communicator 是否被捕获图间接引用；
- inactive expert/peer 是否在所有 kernel 分支中被跳过；
- shrink 发生在 capture 前、capture 中、capture 后分别如何处理。

因此文档只能说“实现选择了地址稳定、面向不重捕图的参数设计”，不能说“Graph 模式无需重捕并已验证”。

## 8. 为什么没有实现透明请求 replay

旧方案曾提出：暂停后提取 in-flight request、retract KV/cache 状态、重排 topology，再把请求重新入队。

当前分支没有采用，原因不是该目标没有价值，而是它会同时扩大以下一致性边界：

- KV cache 已写入位置和释放顺序；
- speculative decoding 中间状态；
- stream 输出已经发送给客户端的 token；
- expert dispatch 已发出但未完成的 token；
- scheduler queue、batch 和 prefix cache 的幂等性；
- 多 scheduler rank 的 barrier 和请求归属。

v0.1.0 使用更明确的语义：受故障影响请求失败/丢弃，survivor 在拓扑提交后处理新请求。透明 replay 若未来实现，应作为独立能力设计和验证，不能夹带在设备适配中。

## 9. 为什么没有引入通用 `FtDeviceOps`

旧文档提出统一 hook：prepare、recover、rebuild、replay 等。当前实现直接在少数 NPU/MC2 分支点执行动作。

对 v0.1.0，这种窄改动有两个现实好处：

- 不要求 GPU 迁移到一个尚未稳定的抽象；
- NPU 的 stop/restart、Gloo/HCCL 组和 internal-format copy 顺序保持可见。

代价是设备判断散落，未来第二个 NPU backend 或更多恢复动作会增加维护成本。只有当至少两条设备路径出现稳定、同构的生命周期后，再提取小而明确的接口更合适。抽象应从已验证事务归纳，而不是先定义一个覆盖所有设想的框架。

## 10. rank0 与 coordinator

当前 GPU 稳定测试避免向 rank0 注入故障，因为 Mooncake coordinator/rendezvous 所有权尚与 rank0 耦合。NPU 分支也没有提供 rank0 特殊恢复的硬件证据。

这应作为后端/部署约束处理：

- 测试注入选择非 rank0；
- API 不必永久编码禁止 0，除非产品决定将该限制变成公共契约；
- 若未来要支持 rank0 shrink，应先把 coordinator/store 生命周期从可移除计算 rank 解耦；
- 文档不能把“测试没覆盖”误写成“设计必然不允许”，也不能反向宣称已支持。

## 11. 决策记录

| 决策 | v0.1.0 结论 | 依据 |
|---|---|---|
| 外部 API 是否设备分叉 | 否 | 运维和 LB 需要统一语义 |
| NPU 是否复制 Mooncake stable-group 实现 | 否 | 当前 MC2/HCCL 未证明 masked old group 可恢复 |
| NPU 是否重建全局 `_TP` | 否 | 实际代码只创建专用 survivor groups |
| 控制面 id 是否 compact | 否 | API/status/日志保持 original DP id |
| MC2 kernel 是否使用 compact id | 是 | `elastic_info` 和 physical expert id 映射 |
| topology tensor 是否换地址 | 否 | 原地 copy，面向 Graph 参数稳定 |
| NPU retry 是否支持 | 否 | 缺 device/group/elastic-info 恢复事务 |
| NPU rejoin 是否支持 | 否 | 缺逆向 membership 与硬件验证 |
| 受故障请求是否透明 replay | 否 | 当前使用 discard/failure 语义 |
| 是否已证明不重捕图 | 否 | 仅有地址稳定单元测试意图 |

## 12. 后续决策点

集成前必须明确：

1. NPU `active_mask` 的规范 rank 空间是 DP 还是 global scheduler rank；
2. 当前版本是否正式限制每 DP 一个 scheduler rank；
3. NPU stop/restart 失败时进程级 fail-stop 和外部重启责任如何划分；
4. compact survivor group 是否允许多次连续 shrink，以及 generation/store 生命周期如何清理；
5. NPU Graph 的产品契约是保持旧图、按 topology 重捕，还是禁用 Graph；
6. rank0 是否是永久不可移除角色，还是应解耦 coordinator；
7. rejoin 是 v0.1.x 必需能力，还是先以进程重启恢复全拓扑。

这些问题在没有代码或硬件证据前应保留为明确决策项，不应由文档替实现作答。
