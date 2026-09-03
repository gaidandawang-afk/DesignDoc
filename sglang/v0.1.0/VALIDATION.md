# SGLang DP-only FT v0.1.0 验证证据与缺口

> 核对日期：2026-08-26
>
> 原则：代码存在、单元测试存在、本地执行通过、GPU/NPU 硬件 E2E 通过是四个不同证据等级

## 1. 证据分级

| 等级 | 定义 | 可用于声明 |
|---|---|---|
| D0 设计 | 文档/接口意图 | 只能描述目标 |
| D1 代码 | 精确分支提交包含实现 | “已实现代码” |
| D2 单元 | 测试在明确环境执行并通过 | 局部逻辑成立 |
| D3 设备 E2E | 冷启动、故障、控制、推理、精度和 artifact 完整 | 指定设备/拓扑受支持 |
| D4 物理故障 | 真实主机下电、网络隔离等 | 物理节点故障受验证 |

本版本 GPU 主要达到 D3，逻辑节点 lease 达到 D3 但物理下电仍未达到 D4。NPU 静态缩容达到 D1；分支含单元测试源码，但本次环境未能重跑，不能把它升级为本次 D2/D3 证据。

## 2. GPU 事实基线

### 2.1 源码与依赖

```text
SGLang workspace: D:\Codex\repos\sglang
branch:           codex/ft-vllm-api-refactor
reviewed HEAD:    90380d6e6ce7bdefc3d209f4fe284623d40720fb

Mooncake workspace: D:\Codex\repos\mooncake-nohca-ft
branch:             codex/mooncake-nohca-ft
reviewed HEAD:      d727290c86e4c821a6fc4c22848ae5c0f269f4f5

server_tool workspace: D:\Codex\tools\server_tool
branch:                codex/test/ft-vllm-api-refactor
reviewed HEAD:         7367ae315f524477abca46c51c9b082658d58fa1
```

GPU artifact 根：

```text
D:\Codex\tools\server_tool\work\sglang-ft-vllm-api-refactor-737-suite-gpu4567\artifacts
```

测试契约索引：

```text
D:\Codex\tools\server_tool\skills\test-service\cases\sglang\INDEX.md
```

### 2.2 分支关键提交

| 提交 | 作用 |
|---:|---|
| `f6b433...` | 初始 FT 能力 |
| `41f28a...` | 对齐 vLLM 风格 FT API |
| `8283227...` | 简化控制消息 |
| `7375c482ba` | 拒绝显式路由到 inactive DP |
| `8a8860da82` | 成功操作也在 status 显示 request id |
| Mooncake `1ca33d50` | deferred recovery 的本地 fast path |
| Mooncake `d727290c` | Graph capture 跳过 inactive peers |

## 3. GPU 场景矩阵

以下 16 个场景来自 server_tool 当前 Index。稳定契约不向 rank0 注入故障。

| # | 场景 | 核心断言 | 结果 |
|---:|---|---|---:|
| 1 | FT 关闭原生 in-flight | 无 FT 干预、请求完成 | 19/19 |
| 2 | continue status-only | 状态查询不改变 topology | 30/30 |
| 3 | pause + scale-down | pause、survivor 恢复、removed dead | 33/33 |
| 4 | exception + retry | unhealthy、retry、路由恢复 | 32/32 |
| 5 | scale-down 后 survivor exception + retry | 连续事务和 request id | 45/45 |
| 6 | exception + scale-down | incident 前置条件、survivor 服务 | 36/36 |
| 7 | A×C>1 whole-DP shutdown | 一个 DP 的全部 scheduler rank 被移除 | 30/30 |
| 8 | continue + in-flight kill | survivor 不暂停、受影响请求结束 | 20/20 |
| 9 | double scale-down | 两次 membership 提交不串代 | 31/31 |
| 10 | continuous 4→3→2→1 | Qwen 冗余 experts、连续收缩 | 43/43 |
| 11 | rejection contracts | envelope/busy/前置条件/inactive route | 41/41 |
| 12 | continue whole-node rejoin | lease、进程、native、route、推理 | 50/50 |
| 13 | pause scale-down rejoin | dead→自动恢复，无 recover API | 62/62 |
| 14 | CUDA Graph rejoin | capture、shrink、恢复、精度、replay | 78/78 |
| 15 | 历史 continue exception-injection 兜底 | 通用 Scheduler 异常包装路径 | 23/23 |
| 16 | unattended pause timeout | 无控制输入时保持隔离并有界结束 | 22/22 |

### 3.1 证据组合限制

场景 1–13、15–16 在 SGLang `7375c482ba` 上完成，共 15/16 场景、517/517 断言。

场景 15 使用一次性 ModelRunner exception injector，验证的是通用 Scheduler 异常兜底，不是 Mooncake membership loss 的 continue 主链路，不能据此把 discard/resume 写成 Elastic EP continue 的外部语义。后者应验证 `forward_raw` 不抛异常、Mooncake active-rank 自动 4→3、EPLB 收敛和同一请求重新推理。

CUDA Graph/status 增量场景在：

```text
SGLang  8a8860da82
Mooncake d727290c
run      status-requestid-cudagraph-d727-r1
result   78/78
```

所以当前可以说“16 个契约场景均有通过证据”，但不能说“在同一个 `8a8860da82 + d727290c` 冷启动套件中一次性重跑了全部 16 个场景”。发布前建议补这一轮，消除相邻提交组合证据。

### 3.2 本地自动测试

此前在当前源码模块隔离环境中执行：

- SGLang FT 相关真实模块测试：31/31；
- server_tool `python -m unittest tests.test_sglang_cases`：35/35。

这些测试验证控制/用例逻辑，不替代 GPU E2E。

## 4. GPU 关键 oracle

### 4.1 apply 成功

POST 只需返回：

```text
HTTP 202
message = Request accepted; poll /fault_tolerance/status for updates.
request_id = submitted request id
```

随后必须等待 status：

```text
last_ft_request_id == submitted request id
ft_error absent
target engine states match operation
```

### 4.2 inactive route

```text
explicit routed_dp_rank points to scaled-down DP
=> HTTP 503
=> error message contains "routed_dp_rank=<N> is not active"
```

HTTP 层 error body 是 OpenAI 风格 `error.message`；不同调用包装器可能投影为 `message` 或 `detail`，测试应在边界层归一化，但不能接受 HTTP 200。

### 4.3 incident 状态

进程被 kill 时，目标 DP 直接成为 `dead`，survivor 仍全部 `healthy`。continue 不经过 `unhealthy`；pause 的 `unhealthy` 用于进程仍存活、等待控制面处置的 incident。

### 4.4 rejoin

通过条件至少包括：

- dead/inactive 阶段请求被拦截；
- 新进程 alive；
- native active mask 恢复；
- pending recovery 清空；
- expected/route 恢复；
- status 为 healthy；
- 显式路由推理和精度通过。

## 5. watchdog 证据边界

### 5.1 已验证

- Manager 记录每个逻辑 node 的 heartbeat lease；
- lease 超过 5 秒后标记它广告的 global ranks down；
- 一个 node 可广告多个 global ranks，单元测试验证 `[2,3]` 一起失效；
- E2E 日志验证逻辑 node 3 租约过期并触发 rejoin 场景。

### 5.2 未验证

- 独立物理主机直接下电；
- 跨机网络单向/双向分区；
- OS 存活但 GPU/NIC/HCA 局部失效；
- heartbeat store/Manager 所在主机故障；
- 多 rank 真实节点上所有 rank 的同时丢失和恢复。

因此 `observe_watchdog_heartbeat` 已验证“整逻辑节点租约聚合场景”，还不能标记为 D4“真实整节点下电”。

## 6. Mooncake Graph 证据

旧组合：

```text
SGLang 7375c482ba
Mooncake 1569df4d
```

观察到 deferred fast path 未准备及：

```text
cudaErrorStreamCaptureUnjoined
```

对照旧成功记录和 Mooncake 差异后，根因定位为依赖配对缺少 Graph capture 前 local fast path 与 inactive peer 跳过，并非 FT API 重构破坏图模式。

修复组合：

```text
SGLang 8a8860da82
Mooncake d727290c
```

CUDA Graph rejoin 场景 78/78，通过 capture-before-native-join、恢复、精度和 replay 断言。

## 7. NPU 事实基线

```text
workspace: D:\Codex\repos\sglang-review-0814-npu-ft-minimal
branch:    codex/review-0814-npu-ft-minimal
HEAD:      127f5a6711f7133597d7ae496577427e0758d037
base:      2abb1d2c37
```

关键提交：

| 提交 | 作用 |
|---:|---|
| `edaee40bf5` | DP-only FT 基线 |
| `2ea9fffb87` | NPU MC2 minimal fault scale-down |
| `127f5a6711` | 为 FT MLP-sync collective 增加有界等待 |

该分支相对当前 GPU/API 分支已经分叉，不能把两边功能表逐项求并集后写成一个已构建二进制。

## 8. NPU 测试现状

### 8.1 分支内测试代码

| 测试 | 预期覆盖 |
|---|---|
| `test_npu_mc2.py` | sparse survivor map、compact expert id、data pointer 稳定 |
| `test_process_group_context.py` | original/compact EPLB rank |
| `test_npu_mlp_sync_timeout.py` | collective error/timeout |
| `test_controller.py` 增量 | NPU+MC2 gate |
| `test_scheduler_control.py` 增量 | interruption 回到 FT loop |

### 8.2 本次执行尝试

执行命令：

```text
python -m pytest \
  test/registered/unit/elastic_ep/test_npu_mc2.py \
  test/registered/unit/eplb/test_process_group_context.py \
  test/registered/unit/fault_tolerance/test_npu_mlp_sync_timeout.py -q
```

收集阶段被本地环境阻塞：

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'sglang'
```

这是测试环境缺依赖，不是用例失败，也不是通过证据。未发现可回取的 NPU 硬件 E2E artifact。

### 8.3 当前 NPU 证据等级

| 能力 | 等级 | 说明 |
|---|---|---|
| static scale-down 数据面代码 | D1 | stop/restart、group、EPLB、elastic-info 已提交 |
| original/compact 映射 | D1 | 有单测源码，本次未执行 |
| MLP-sync timeout | D1 | 有单测源码，本次未执行 |
| NPU static scale-down E2E | 未达到 D3 | 无硬件 artifact |
| NPU Graph shrink/replay | 未达到 D3 | 仅 tensor 地址稳定意图 |
| retry/continue/rejoin | D0/未支持 | 代码事务不完整 |
| 最新异步 API 兼容 | 未集成 | 仍使用旧同步 API |

## 9. NPU 硬件验证计划

每次运行使用独立 run/artifact 目录，至少记录源码提交、环境、NPU 型号、驱动/CANN、MC2 版本、端口、进程和完整日志。

### 9.1 冷启动基线

- 验证实际导入路径和包版本；
- 证明 `device=npu`、`elastic_ep_backend=mc2`、EPLB、DP attention 配置生效；
- 记录 original DP/EP/global rank 映射；
- status 和显式路由基线；
- 精度/输出基线。

### 9.2 静态缩容

1. 只向非 rank0 注入可控 incident；
2. 确认集群进入 pause；
3. 提交 scale-down；
4. 观察 survivor `stop_device/restart_device`；
5. 记录 membership/generation、Gloo barrier、HCCL warmup；
6. 记录 original↔compact 映射和 elastic-info；
7. 验证 EPLB 强制 rebalance 和 expert placement；
8. 验证 removed route 被拒绝、survivor route 成功；
9. 验证精度和持续请求；
10. 重复冷启动，确认没有依赖 stale store/group。

### 9.3 失败测试

- survivor 缺席导致 group 创建超时；
- MLP-sync old group 卡住后 5 秒内进入 FT loop；
- stop/restart 抛错；
- HCCL warmup 失败；
- invalid active mask/rank 空间不一致；
- 连续 4→3→2 shrink；
- Graph 打开时 capture 前/后 shrink。

## 10. 待决代码问题

### 10.1 NPU `attn_tp>1` global-rank 小改动

结论：`active_mask` 的规范空间应保持为 global scheduler rank，不需要压缩成 DP-level mask。Manager 已按 whole DP 展开 mask，所有 survivor attention-TP sibling 都会收到并执行同一命令，MC2/EPLB 也使用 global EP/rank 空间。

核对提交仍有两处局部问题：

- `init_npu_ft_communication()` 使用 `dp_rank/dp_size`，应改为 `tp_rank/tp_size`；
- NPU MLP-sync gather 写到 `[active_original_ranks, 0]`，应写到 `global_info_tensor` 展平后的 global-rank 槽位。

预计产品代码改动为 3–6 行、少于 10 行。该结论把问题从“规格限制”修正为“设计支持但当前提交缺少小适配”。完成标准是：补 DP2×ATTN_TP2 单元测试，并在 NPU 上验证 whole-DP shrink、MLP-sync、EPLB、精度和持续请求；`attn_cp>1` 另行验证。

### 10.2 `scale_down` instruction-specific 参数校验

GPU 当前 `validate_apply_payload()` 只检查 body、instruction、`params` 类型和 request id 类型；后台任务直接读取 `params["removed_dp_ranks"]`，也没有在提交前完整校验元素类型、范围和负数。

结果是部分畸形请求可能先返回 202，再因 `KeyError`/索引异常进入 fatal task wrapper，而不是得到稳定的 400 或聚合 `ft_error`。这属于 API 防御性校验缺口，应由产品决定：

- 在 submit 前同步校验并返回 400；或
- 在后台将语义错误转换为本次 request id 的 `ft_error`；
- 不应让不可信 HTTP 参数触发控制面 fail-stop。

### 10.3 空 request id

当前 request id 只要求是字符串，省略时默认为 `""`。这符合当前代码，但会削弱跨操作相关性。建议在后续契约中要求调用方提供非空唯一值，或由服务端在空值时生成 UUID；在代码改变前，测试不能误写成“缺 request id 返回 400”。

### 10.4 continue 的 route/status rejoin gate

SGLang `90380d6e6c` 将 continue 的 route 和 status 统一建立在 observed-ready mask 上：DP 的全部进程存活、runtime active，且 `pending_recovery_global_ranks` 中不存在该 DP 的成员。进程重新注册但尚未收到新的 runtime-ready 上报时，status 保持 `dead`，route 保持关闭。

本地隔离测试覆盖 runtime-only、process-first、runtime-first、expected 不变以及已 scale-down DP 的 rejoin，共 26/26 通过。该变更尚未在 GPU E2E 上按精确 HEAD 重跑，因此证据等级为 D2，不替代现有 D3 场景证据。

### 10.5 全量精确 HEAD 重跑

GPU 16 场景证据跨 `7375c482ba` 和 `8a8860da82`。若 v0.1.0 要作为发布基线，应在单一 SGLang/Mooncake 精确提交上重跑全部场景并形成一个 summary artifact。

## 11. 文档验收规则

文档更新完成后应执行：

- 所有相对 Markdown 链接目标存在；
- 不再引用 `disabled` 或显式 `recover` 作为当前契约；
- 旧同步 API 只出现在“历史/分支差异”语境；
- GPU/NPU 每个支持结论都绑定设备和证据等级；
- `git diff --check` 无空白错误；
- Git 只包含本次迁移的 Markdown，不包含 `.idea`、wheel、drawio 或本地 `sglang/mooncake` 文件；
- 提交信息说明“为何重写”和审阅的精确分支基线。
