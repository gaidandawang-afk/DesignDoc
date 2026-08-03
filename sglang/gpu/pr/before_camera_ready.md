可以。完整计划如下，我建议你按这个执行：**PR1 不等模型切分完整重构；但必须并行做一个基于新框架的 2+2 spike，验证后续不会推翻。**

核心原则是：

```text
ft_preview_refactor / ft_preview_pr_ready
  负责当前社区 PR：effective TP=1 FT 控制面 + topology-aware guard

ft_dev
  只作为旧框架上的模型切分行为参考，不直接重构、不直接 merge

ft_rankspace_spike
  基于新框架重新做最小 2+2 vertical slice，验证 rank-space seam

ft_rankspace_next
  spike 成功后，正式承接后续 TP>1 / 模型切分支持
```

设计文档本身已经给了 PR1 的边界：对外接口只有 `status/apply`，对外只暴露 DP rank，v5 初版只支持 effective TP=1，而且不能按 raw `--tp-size > 1` 粗暴拒绝启动。([GitHub][1]) 另外，`ft_preview_refactor...ft_dev` 当前已经是 8 commits / 17 files changed，并且第一笔核心提交已经涉及 physical scheduler lifecycle、global/EP rank-space、ACK、process replacement、direct-control socket、scale_down shutdown/non-shutdown 等内容，这说明它已经是后续 rank-space 重构 PR 的体量，不适合塞进首个社区 PR。([GitHub][2])

---

## 总体分支计划

```text
ft_preview_refactor
  当前新框架基线，不直接污染

ft_preview_pr_ready
  基于 ft_preview_refactor
  目标：当前社区 PR

ft_dev
  旧框架模型切分参考实现
  目标：行为 oracle / 测试用例来源 / 设计参考

ft_rankspace_spike
  基于 ft_preview_pr_ready 或 ft_preview_refactor
  目标：验证新框架能否承接 2+2 rank-space

ft_rankspace_next
  基于最终 PR1 分支
  目标：正式开发 PR2/PR3
```

推荐本地目录：

```text
D:\Codex\repos\sglang-pr-ready
D:\Codex\repos\sglang-ftdev-read
D:\Codex\repos\sglang-rankspace-spike
```

命令：

```bash
cd D:\Codex\repos\sglang

git fetch origin

git worktree add ../sglang-pr-ready ft_preview_refactor
cd ../sglang-pr-ready
git checkout -b ft_preview_pr_ready

cd ../sglang
git worktree add ../sglang-ftdev-read ft_dev

cd ../sglang
git worktree add ../sglang-rankspace-spike ft_preview_refactor
cd ../sglang-rankspace-spike
git checkout -b ft_rankspace_spike
```

---

# 阶段 0：冻结事实，不急着改代码

目标：先把三个分支的职责钉死。

## 0.1 在 `ft_preview_pr_ready` 记录当前 PR1 基线

执行：

```bash
cd D:\Codex\repos\sglang-pr-ready

git branch --show-current
git rev-parse HEAD
git status --short

git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

产出一个文件：

```text
docs/ft_pr_ready_notes.md
```

里面记录：

```text
PR1 branch: ft_preview_pr_ready
Base: ft_preview_refactor
Scope: effective TP=1 FT control plane
Non-goal: TP>1 / model-sharded FT
```

## 0.2 在 `ft_dev` 上只做代码考古

执行：

```bash
cd D:\Codex\repos\sglang-ftdev-read

git branch --show-current
git rev-parse HEAD
git status --short

git diff --stat ft_preview_refactor...ft_dev
git diff --name-status ft_preview_refactor...ft_dev
git log --oneline --reverse ft_preview_refactor..ft_dev
```

产出一个表：

```text
docs/ft_dev_archaeology.md
```

格式：

```text
commit:
changed files:
solves:
rank-space concept:
depends on:
PR1 candidate: yes/no
PR2 candidate: yes/no
PR3 candidate: yes/no
old-framework workaround: yes/no/unknown
```

这个阶段的规则：

```text
不 cherry-pick
不 rebase
不重构 ft_dev
不把 ft_dev 当成可迁移代码
```

---

# 阶段 1：先把当前社区 PR 做干净

目标：`ft_preview_pr_ready` 可以独立提社区 PR。

PR1 的名字建议是：

```text
feat(ft): add Mooncake fault-tolerance control plane for effective-TP=1
```

或者更保守：

```text
feat(ft): add DP-rank fault tolerance control plane for Mooncake
```

## 1.1 PR1 明确支持范围

PR1 支持：

```text
FT enabled
Mooncake elastic EP
DP-rank status
GET /fault_tolerance/status
POST /fault_tolerance/apply
retry
scale_down
FT_pause
FT_continue
kill / exception / inactive_rank 基础语义
DPC routing 避开 dead DP rank
FT disabled parity
effective TP=1
```

PR1 不支持：

```text
effective TP>1
模型切分 FT
physical/global rank public API
rank replacement
rejoin
direct-control by global rank
process registry
NCCL compact-rank rebuild
PP>1
Ray
PD disaggregation
NPU
multi-tokenizer worker
```

PR 描述必须写清楚：

```text
This PR intentionally does not enable model-sharded FT.
It only centralizes topology validation so follow-up TP>1 support does not need to reinterpret raw --tp-size.
```

## 1.2 PR1 允许吸收的模型切分预埋

只允许吸收这几类：

```text
1. minimal topology helper
2. centralized supported-config check
3. effective_tp / ranks_per_dp 判断
4. unsupported reason
5. focused unit tests
6. 文档说明后续 rank-space refactor
```

推荐新增或整理：

```text
python/sglang/srt/fault_tolerance/topology.py
python/sglang/srt/fault_tolerance/controller.py
test/srt/test_fault_tolerance_topology.py
test/srt/test_fault_tolerance_controller.py
```

核心数据结构只要这样就够：

```python
@dataclass(frozen=True)
class FaultToleranceTopology:
    dp_size: int
    global_rank_count: int
    ranks_per_dp: int

    def dp_members(self, dp_rank: int) -> list[int]:
        start = dp_rank * self.ranks_per_dp
        return list(range(start, start + self.ranks_per_dp))

    @property
    def effective_tp(self) -> int:
        return self.ranks_per_dp
```

PR1 的 guard 逻辑：

```python
topology = resolve_ft_topology(server_args)

if topology.effective_tp != 1:
    return Unsupported(
        reason="ft_unsupported_effective_tp_gt_1",
        message=(
            "Fault tolerance currently supports only effective TP=1. "
            "Raw --tp-size is not used directly because DP-attention/Mooncake "
            "may encode global rank space in tp_size."
        ),
    )
```

注意：PR1 里这个 topology helper 可以存在，但不要让它驱动 runtime 的 global-rank command path。

## 1.3 PR1 禁止吸收的代码

这些不要进 `ft_preview_pr_ready`：

```text
process_registry.py
direct-control socket
global-rank FT command routing
global-rank ACK barrier
replacement PID
rank rejoin output
ActiveRanksOutput.rank_space 全链路改造
physical state -> DP projection runtime
scheduler global-rank identity 改造
same-DP double kill 运行时逻辑
```

原因很简单：这些一进 PR1，维护者会按 TP>1 / rejoin / rank replacement 来 review，你的首个 PR 会膨胀。

## 1.4 PR1 单测矩阵

至少补这些 unit tests：

```text
test_resolve_topology_dp1_tp1
test_resolve_topology_dp4_tp4_dp_attention_is_effective_tp1
test_resolve_topology_dp2_tp4_dp_attention_is_effective_tp2
test_ft_supported_effective_tp1
test_ft_reject_effective_tp_gt_1
test_ft_reject_pp_gt_1
test_ft_reject_ray_if_currently_unsupported
test_ft_reject_pd_if_currently_unsupported
test_ft_reject_npu_if_currently_unsupported
```

关键 case：

```text
tp=4 dp=4 enable_dp_attention=true
  expected: supported, because ranks_per_dp=1

tp=4 dp=2 enable_dp_attention=true
  expected: unsupported_effective_tp_gt_1

tp=1 dp=4
  expected: supported

tp=2 dp=1
  expected: unsupported_effective_tp_gt_1
```

## 1.5 PR1 验证命令

```bash
python -m py_compile \
  python/sglang/srt/fault_tolerance/controller.py \
  python/sglang/srt/fault_tolerance/topology.py \
  python/sglang/srt/managers/tokenizer_manager.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/data_parallel_controller.py \
  python/sglang/srt/managers/io_struct.py

python -m pytest \
  test/srt/test_fault_tolerance_topology.py \
  test/srt/test_fault_tolerance_controller.py \
  test/srt/test_fault_tolerance_commands.py \
  -q

python -m ruff check python/sglang/srt/fault_tolerance test/srt/test_fault_tolerance_*.py
git diff --check
```

GPU smoke 建议记录：

```text
noFT parity
FT disabled parity
FT_pause + kill + retry
FT_pause + kill + scale_down
FT_continue + kill
inactive_rank routing
status/apply API
```

---

# 阶段 2：并行做新框架 2+2 spike

目标：验证新框架是否能承接模型切分，不追求可合入。

这个分支是：

```text
ft_rankspace_spike
```

它回答一个问题：

```text
在 ft_preview_refactor 新控制面下，支持 TP=4 DP=2 DP-attention 的最小 rank-space seam 是什么？
```

## 2.1 spike 第一目标：只做状态和 mask 投影

目标拓扑：

```text
tp_size=4
dp_size=2
ep_size=4
attn_cp_size=1
enable_dp_attention=true

global ranks: [0,1,2,3]
DP0 members: [0,1]
DP1 members: [2,3]
```

先只实现这四个函数：

```python
dp_members(dp_rank) -> list[global_rank]

global_rank_to_dp_rank(global_rank) -> dp_rank

physical_states_to_public_dp_states(states) -> list[DPState]

physical_states_to_dp_route_mask(states) -> list[bool]

physical_states_to_ep_active_mask(states) -> list[bool]
```

语义：

```text
kill global rank 2

physical:
  rank0 healthy
  rank1 healthy
  rank2 dead
  rank3 healthy

public status:
  DP0 healthy
  DP1 dead

EP mask:
  [1,1,0,1]

DP route mask:
  [1,0]
```

这个阶段先不碰 rejoin。

## 2.2 spike 第一批文件

允许改：

```text
python/sglang/srt/fault_tolerance/topology.py
python/sglang/srt/fault_tolerance/rank_space.py
python/sglang/srt/fault_tolerance/controller.py
test/srt/test_fault_tolerance_topology.py
test/srt/test_fault_tolerance_rank_space.py
test/srt/test_fault_tolerance_controller.py
```

不碰：

```text
process_registry.py
direct-control socket
scheduler command path
tokenizer rejoin path
DPC actual routing path
Mooncake rejoin path
```

## 2.3 spike 第一验收

unit tests：

```text
topology:
  tp=4 dp=2 attn_cp=1 -> global_rank_count=4, ranks_per_dp=2
  DP0 -> [0,1]
  DP1 -> [2,3]
  global 2 -> DP1

projection:
  all healthy -> public [healthy, healthy]
  rank2 dead -> public [healthy, dead]
  rank2 dead -> ep [1,1,0,1]
  rank2 dead -> dp_route [1,0]
  ranks2,3 dead -> public [healthy, dead]
  ranks2,3 dead -> ep [1,1,0,0]
  ranks2,3 dead -> dp_route [1,0]
```

如果这些都写不顺，说明新框架 rank 边界还没找准，不应该迁 `ft_dev` 代码。

---

# 阶段 3：spike 第二步，接入最小 runtime vertical slice

目标：不是完整支持 TP>1，而是证明新框架能跑通最小 2+2 控制面。

## 3.1 接入 TokenizerManager / DPC 的 mask 分离

新增或改造：

```text
ActiveRanksOutput
  rank_space: "ep" | "dp_route"

TokenizerManager:
  给 Mooncake / worker 发 ep_active_mask
  给 DPC 发 dp_route_mask

DPC:
  只接受 dp_route mask
```

核心规则：

```text
EP/global mask 是 physical/global rank 维度
DPC route mask 是 DP rank 维度
public status 是 DP rank 维度
```

## 3.2 接入最小故障路径

先只做 kill，不做 exception，不做 rejoin：

```text
kill global rank 2
  controller record global rank 2 dead
  EP mask -> [1,1,0,1]
  DP route mask -> [1,0]
  status -> DP1 dead
  DP0 request 200
  DP1 directed request 503 / unroutable
```

## 3.3 spike 第二验收

remote-agent 或手工验证：

```text
B0 baseline:
  DP0 routed generation 200
  DP1 routed generation 200
  EP mask [1,1,1,1]
  DP route mask [1,1]

F1 rank2 kill/continue:
  kill global rank 2
  public status DP1 dead
  EP mask [1,1,0,1]
  DP route mask [1,0]
  DP0 generation 200

M1 sibling retention:
  rank3 still alive
  rank3 still EP-active
  but DP1 is not routable
```

不要在这个阶段做：

```text
pause/retry
scale_down shutdown
same-DP double kill
rejoin
replacement PID
direct-control
```

---

# 阶段 4：spike 结束后的决策点

到这里再决定 PR1 是否吸收 spike 的东西。

## 4.1 允许反向吸收到 PR1 的内容

只有这些可以回到 `ft_preview_pr_ready`：

```text
更准确的 topology helper
更清楚的 unsupported reason
不会改变 runtime 行为的 rank-space 命名
focused tests
文档说明
```

例如：

```text
FaultToleranceTopology
resolve_ft_topology()
is_supported_ft_topology()
unsupported_effective_tp_gt_1
```

## 4.2 不允许回到 PR1 的内容

这些仍然留在 `ft_rankspace_next`：

```text
ActiveRanksOutput.rank_space
physical controller state
physical -> DP projection runtime
EP mask / DP route mask runtime split
DPC mask change
scheduler global-rank identity
process registry
rejoin
direct-control socket
```

判断标准：

```text
如果这段代码改变当前 effective TP=1 runtime 行为，不进 PR1。
如果这段代码只是让 unsupported 判断更准确，可以进 PR1。
如果这段代码是为了 TP>1 能运行，不进 PR1。
```

---

# 阶段 5：整理并提交 PR1

目标：社区只 review 一个干净的 DP-rank FT 控制面。

## 5.1 PR1 commit 结构

建议拆成 4 到 6 个 commits：

```text
1. feat(ft): add fault-tolerance status/apply API
2. feat(ft): implement DP-rank FT controller and apply flow
3. feat(ft): integrate FT pause/continue with tokenizer and scheduler
4. feat(ft): update DPC routing for inactive ranks
5. refactor(ft): centralize topology support checks
6. test(ft): add focused FT controller/topology tests
```

不要把 topology 放在最后的“顺手修”。
它应该是一个明确 commit，因为它解释了为什么不是 raw `tp_size` 判断。

## 5.2 PR 描述模板

```text
## Summary

This PR adds the first version of SGLang Mooncake fault-tolerance control plane.

It supports DP-rank level fault handling for effective TP=1:
- GET /fault_tolerance/status
- POST /fault_tolerance/apply
- retry
- scale_down
- FT_pause / FT_continue behavior
- inactive-rank routing update

## Scope

Supported:
- Mooncake elastic EP
- DP-rank public status
- effective TP=1

Not supported in this PR:
- model-sharded FT / effective TP>1
- rank replacement / rejoin
- physical/global-rank public API
- PP>1
- Ray / PD / NPU

## Topology note

This PR intentionally does not reject raw --tp-size > 1 directly.
In DP-attention/Mooncake configurations, raw tp_size may represent the global rank space rather than effective model-sharding TP.

The support check is centralized through normalized FT topology and currently rejects only effective TP>1.

## Validation

- py_compile focused FT modules
- pytest focused FT tests
- ruff / diff check
- noFT parity smoke
- FT disabled parity smoke
- FT_pause kill + retry
- FT_pause kill + scale_down
- FT_continue kill
- inactive-rank routing
```

## 5.3 PR1 review 防守口径

维护者问：为什么不支持 TP>1？

答：

```text
TP>1 requires separating public DP-rank status/routing from internal physical/global-rank fault control, ACK, EP mask, and process registry. That is a larger rank-space refactor and will be submitted separately.
```

维护者问：为什么加 topology helper？

答：

```text
Because raw --tp-size is not a reliable indicator of effective model-sharding TP in DP-attention/Mooncake configurations. The initial PR only centralizes the support check and keeps runtime behavior DP-rank based.
```

维护者问：这个 PR 会不会影响非 FT？

答：

```text
FT code is gated by --enable-fault-tolerance. FT disabled and noFT parity are part of validation.
```

---

# 阶段 6：PR1 后正式开 PR2：rank-space refactor

目标：把 spike 变成可合入实现。

基线：

```bash
git checkout ft_preview_pr_ready
git checkout -b ft_rankspace_next
```

PR2 范围：

```text
支持 TP=4 DP=2 DP-attention 下的基础 kill / pause / retry / scale_down
不支持完整 rejoin
```

## 6.1 PR2 核心改动

可以引入：

```text
rank_space.py
ActiveRanksOutput.rank_space
physical/global rank states
public DP projection
EP mask vs DP route mask separation
DPC route mask update
scheduler ACK identity as global rank
```

## 6.2 PR2 验收矩阵

```text
B0:
  baseline DP0/DP1 generation 200

F1:
  kill global rank 2
  DP1 public dead/unroutable
  DP0 generation 200

F2:
  rank2 pause/retry
  DP0 restored
  DP1 remains dead/unroutable unless full DP group restored

M1:
  sibling rank3 EP-active retention
  EP mask [1,1,0,1]
  DP route mask [1,0]

S1:
  scale_down shutdown=false
  rank3 remains EP-active

S2:
  scale_down shutdown=true
  rank2/rank3 both removed
  EP mask [1,1,0,0]

F3:
  same-DP double kill
  ranks 2 and 3 dead
  DP1 dead
  DP0 generation 200
```

---

# 阶段 7：PR3 再做 rejoin / direct-control

目标：把 `ft_dev` 里最复杂的部分正式产品化。

PR3 才引入：

```text
process_registry.py
replacement PID
direct-control by global rank
multi-launcher direct endpoint
rejoin readiness barrier
same DP group readiness
repeated rejoin
```

## 7.1 PR3 验收矩阵

```text
R1:
  kill rank2
  replacement PID registered
  rank2 rejoins
  DP1 only reopens after ranks [2,3] attention-ready
  DP1 generation 200

R2:
  repeated rejoin
  第二轮不能命中第一轮旧日志 marker
  每轮使用 fresh log 或 byte offset

R3:
  multi-launcher direct-control
  command by global rank reaches correct scheduler

R4:
  replacement process failure
  liveness polling reports failure
  public DP status remains safe
```

---

# 最终执行清单

你现在直接按这个顺序做：

```text
1. 建三个 worktree
   - sglang-pr-ready
   - sglang-ftdev-read
   - sglang-rankspace-spike

2. 在 ft_preview_pr_ready 上收敛 PR1
   - topology-aware guard
   - unsupported effective TP>1
   - focused tests
   - PR 描述
   - smoke evidence

3. 在 ft_dev-read 上只做考古
   - commit 表
   - 文件职责表
   - 哪些是本质 rank-space 问题
   - 哪些是旧框架 workaround

4. 在 ft_rankspace_spike 上重建最小 2+2 seam
   - topology
   - rank_space
   - physical -> DP projection
   - EP mask vs DP route mask
   - kill rank2 minimal path

5. spike 成功后，只把 topology/support check 级别的稳定抽象反向吸收到 PR1

6. 提 PR1

7. PR1 稳定后，基于 PR1 开 ft_rankspace_next，正式做 PR2

8. PR2 稳定后，再做 PR3 rejoin/direct-control
```

---

## 最重要的红线

```text
不要先完整重构 ft_dev。
不要把 ft_dev 当成可直接迁移源。
不要让 PR1 宣称支持 TP>1。
不要把 process_registry/direct-control/rejoin 塞进 PR1。
不要用 4×TP1 证据证明 2+2。
不要用 noFT rejoin 证据证明 FT 控制面。
不要用 raw --tp-size 判断是否支持 FT。
```

你真正要做的是：

```text
PR1：把当前 FT v5 控制面干净合进去。
Spike：证明新框架能容纳 2+2 rank-space。
PR2：正式 rank-space refactor。
PR3：正式 rejoin / direct-control。
```

这个顺序最稳，也最不容易把当前社区 PR 做炸。

[1]: https://raw.githubusercontent.com/gaidandawang-afk/DesignDoc/main/sglang/gpu/v5/Spec/sglang_fault_tolerance_design_v5_arch_zh.md "raw.githubusercontent.com"
[2]: https://github.com/gaidandawang-afk/sglang/compare/ft_preview_refactor...ft_dev "Comparing ft_preview_refactor...ft_dev · gaidandawang-afk/sglang · GitHub"
