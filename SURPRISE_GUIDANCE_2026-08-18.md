# SURPRISE walkthrough — curriculum×MaxRL: current position + next steps

*Received from the PI on 2026-08-18. Saved verbatim; the working plan derived
from it is in the section after the guidance.*

---

# 用 SURPRISE 走查 curriculum×MaxRL：现状定位 + 下一步

先说结论：这个项目本来就在按这套方法论运转（预注册、公开撤回、deletion 实验、identity path 都在）。所以这次调用的价值不在补流程，而在两个发现：**你们的三个负结果其实是同一个坐标错误的三个投影**，以及**瓶颈已经迁移，账本需要重写**。下面按 skill 的环走一遍，最后给出规定的七件套输出。

## 现状体检（五字诀）

- **换坐标 ✓ 但没换到底。** "外部难度启发式 → 估计器系数代数"是论文核心，成立。但残余错误是：A_N(p) = 2(pass@N − pass@1) 度量的是 **signal availability**（估计器在该任务上能发出多少梯度质量），却被部署为 **training utility**（采样应瞄准的对象）。你们在 §8 已经防御性地写了"estimator-side diagnostic, not a theorem of learning progress"——skill 的作用是把这句限制性声明转成下一篇的正面研究对象。
- **留直路 ✓。** floor、estimator 不动、gate≡1→上游 PLR、λ_relabel=0→纯 MaxRL，归因链完整。
- **删假设 ◐。** 已删 adaptive T、LP teacher、随机 relabel 方向（5/5）；还剩四个"理所当然"没删（见 deletion ladder），其中 floor 删除是对你们自己 safety 头条的压力测试。
- **测瓶颈 ✓ 但账本过期。** 65% dead group 是当年的真瓶颈，已被解决；oracle-tie 和 frontier-heavy 归因控制（0.931 vs 0.928）说明 allocation 饱和。
- **押未来 ◐。** peak-location 的赌注押了、输了、诚实报了——这是好研究。但 LLM 侧的未来测试集还薄（1-of-2 seeds、proxy 指标）。

## S — 要攻击的新常识

> "课程分数的价值在于**分配**：更准的难度/信号估计 ⇒ 更好的课程；分数应在部署 N 的峰值处达峰；分数可以原位替换任何 priority。"

谁信、在哪出现：SFL 的 p(1−p)、ADARFT 难度带、PLR 的 MaxMC 线——**以及你们 v1 的 abstract**（peak-location 曾是你们对 p(1−p) 文献声称的唯一进步）。

它预言什么，现状如何：(a) oracle > cheap posterior + hindsight → 被反驳（0.8885 vs 0.8895）；(b) u16 > u64 当部署 N=16 → 被反驳（argmax@u64，Spearman +.93 一路上升）；(c) activity 可替换 MaxMC → 被反驳（.500/.551 < 对照 .539）。三条预言全部被你们自己的数据击穿，所以要正面证伪的常识浓缩为一句：**"availability 即 utility。"**

## U — 更新后的 bottleneck ledger

| 排序 | 层 | 证据 | 判定 |
|---|---|---|---|
| 1 | **评估** | Countdown 只有 bootstrap proxy、无 per-seed raw；E2c 被 gate 7 结构性关闭（5.79%>5%） | 现役第一瓶颈：mean↑/coverage↓ 是真现象还是 proxy 伪影，决定整条 recycling-gate 研究线的生死 |
| 2 | **数据（LLM）** | 3,200 draws / 7,473 prompts ≈ 0.4 visits/prompt；posterior 知道（ρ=−.17, p≈10⁻¹⁷）但无法行动；3-tier 可行 | 挡住 channel-1 向 LLM 迁移；对症药正是 E-LLM-3 的连续 dial + kernel posterior |
| 3 | **机制（safety）** | 只有 GRPO/MaxRL 两点对照；easy-band 定位仅 suggestive（10/12，CI 含 0） | "inversion 维护 easy prompts"未被单变量隔离 |
| 4 | 采样/分配 | hindsight 开启后 teacher 增量：moderate 池 +.005（.878→.883）、frontier-heavy ≈0；例外是 gym 二值（.958→1.000） | **已饱和，从主攻方向划掉**（gym regime 除外） |
| 5 | 优化 | adaptive T 负结果：N=16–32 方差不 binding | 非瓶颈 |

## R — 重参数化：一个因子化统一三个负结果

新对象一：**priority(x) ≈ A_N(p_x) · C(x; 池结构 G, θ)**。A_N 是估计器侧精确项（你们已有），角色从"效用"降级为"可行域/门"——没有 availability 就没有 utility，但有 availability 不等于有 utility。C 是池结构侧的 compounding 项，你们的两行 ODE 已经是它的初版模型（γ 效应在共享技能池上定量重现、flat 池上消失）。三个负结果在此坐标下各归其位：

1. **u64 > u16**：结构化池上 utility 的峰被 compounding 拉向 availability 峰的更难侧——"harder-peaked helps"不再是经验规则，而是 C 项的可预测后果；
2. **AMaze**：availability 当 priority 会被一个 Bernoulli/visit 饿死；当 gate 罩住 per-timestep 富信号就恢复大半（.590）——它的正确角色是门，不是效用；
3. **oracle-tie**：完美的 availability 信息到 sampler ceiling 为止，而 hindsight 改的是 **A 的定义域**（把 u=0 处变成 u>0），所以在 oracle 之上仍 +.005。

新对象二（recycling 通道）：**策略耦合的目的地测度 q_rec(x′|θ_t)**。E2c 的"源多样性随策略变锐而塌缩"不是 nuisance，是这个对象的第一条动力学观测；Countdown 的 mean↑/proxy↓ 是它的第一个候选效应（待原始测量裁决）。gate 只是这个对象上的一个投影控制。

## P — identity path

统一写成 residual 采样器 **q_φ(x) ∝ ρ(x)·exp(β·Û(x))**：β=0 精确退化到 ρ（uniform+floor 基线）；Û = log u_N 退化到现有 teacher（β 即 γ）；C≡const 时整体退化到现方法；gate≡1 → 上游 PLR；λ_relabel=0 → 纯 MaxRL。新论文的每个部件都有 no-op 点。

## R — deletion ladder（剩余四项，一次删一个、给足预算）

1. hindsight 开启时删 teacher：三个 regime 的点已有（+.005 / ≈0 / +.042），补成正式 regime map 发表；
2. 删 Beta+Thompson → batch 经验 pass-rate + softmax(γ)：若打平，"zero difficulty hyperparameters"主张变得更硬；
3. 删 uniform floor（MaxRL+teacher 下）：若 pass@k 塌，safety 主张须改写为"MaxRL**+replay** 安全"；若不塌，主张升级。半天成本，务必自己先知道；
4. 删"MaxRL 特殊性"：RLOO（质量 2p(1−p)，不倒挂）+ frontier teacher，把两点对照变成沿"倒挂程度"的三点单调测试。

## I — prediction ledger：三个能推翻主张的实验

| 实验 | 预期方向 | 机制解释 | 若相反意味着 | 推翻什么 |
|---|---|---|---|---|
| ① branch-and-continue utility 审计（CPU skill-chain：稀疏测真 U_H = J(Train_H(θ;x)) − J(θ)，比较预测子 u_N / p(1−p) / u64 / A·C） | A·C 排序 U_H 显著优于纯 u_N，且优势只在结构化池出现、flat 池消失（ODE 给定量预测） | compounding：前沿一步解锁下一环 | 若 u_N 已把 U_H 排到顶 → C 项经验惰性，factorization 不必要，收缩回"harder-peaked"经验规则 | 新论文的整个 thesis |
| ①b 配套 sampler 级 2×2：{score-N 16/64}×{γ 1/4}×{structured/flat} | 若可互换 → 全归温度；若不可互换 → 峰位独立起效 | 区分 sharpness 与 peak-shift 两条通道 | 两个结局都有信息量 | "harder-peaked"的成分归属 |
| ② RLOO + frontier teacher（maze，与 H6 同 harness 同预算） | 不塌或轻微（无倒挂 ⇒ 无 easy-prompt 维护依赖） | inversion-maintenance | 若塌得像 GRPO → 机制错，safety 降级为纯 estimator-conditioned 排序 | channel-3 的机制解释 |
| ③ raw-outcome recycling 重测（E2c′：冻结外部 probe set + 保留 per-seed 二值结果算真 pass@16；把"测量问题"与 replay 因果控制**解耦**，绕开结构性漂移——测量本身只需 recycling on/off 两臂） | 若 concentration 真：mean@16↑ 而真 pass@16 ↓或平 | 目的地集中于 p≈1 | 若真 pass@16 也↑ → 权衡是 proxy 伪影，gate 计划整体撤销 | recycling-concentration 现象及其上的全部 gate 工作 |

## S — scale & stress 四轴

N 轴（ordering 已验，补 C 随 N 的行为）；池结构轴（7k-row → binned → 连续 dial，这才是 LLM 侧真正的 scaling 轴）；模型轴（360M 的 1-of-2 seeds 至少补到 3 seeds + 一个 ~1B 点，报斜率不报单点）；信号丰富度轴（Bernoulli/visit → MaxRL-group 学生 → per-timestep critic——这正是你们自己点名的 AMaze 后续"score and learner share an algebra"）。

## E — future test matrix

| regime | teacher 增量 | hindsight 增量 | GRPO+teacher 安全？ | ODE 预测最优 sharpness |
|---|---|---|---|---|
| flat 池 | ≈0 | 小 | — | γ→1（预测 γ 效应消失） |
| structured chain | 小（饱和） | 大 | 塌（已证） | γ≈4 附近，由技能图定量给出 |
| frontier-heavy (p≤10⁻⁵) | 0（无信号可分） | 点火 0→0.98 | — | 无定义（A≡0） |
| mastered-heavy | +（退休维护） | ≈0 | 塌更狠（H6） | — |
| reasoning-gym 连续 dial | ? 预注册 | ? | ? | kernel posterior 版本 |
| Hopper / dense（MAZE-SCORE） | via gate | n/a | ? | 富信号下 gate 形态 |

两个真正的"未来测试集"：reasoning-gym（对手设定的 threshold 基线 +13–40 点，无阈值 kernel teacher 去 match-or-beat）和 Hopper（neural、dense）。机器人方向按 skill 补一条**质量门**：binary success 必须先过 tracking/jerk/energy 门再进 Bernoulli，否则课程会学会"用抖动换成功"——RA-L 版本的必要设定。另外每格同时报三列（AUC 速度、最终能力、pass@k 覆盖）：你们已经吃过 token entropy 看不见分布损伤的亏，也别让 0.653→0.878 的速度优势被误读成 0.966→0.984 的最终能力优势。

## 时间线

**一个月内（→9/18 abstract、9/25 paper）**：守住 MAZE-SCORE 48-block（你们 power memo 的最高价值项）和 gated-MaxMC 收尾；加两个便宜高杠杆项——CPU 的实验①+①b（数天，能把"peak-location not supported"从让步小节改写成机制小节：**估计器钉住 ordering，池结构决定 sharpness**，这比原主张更强）和 floor 删除（半天）；GPU 有余量就上 RLOO 单臂。E2c′ 的测量基建放投稿后——abstract 现有的"motivating pass@k recomputed from retained raw outcomes"框定已经诚实，deadline 前别搭基础设施。

**投稿后主线**：availability×utility 论文（对象：U_H 因子化 + q_rec 动力学），部署面走 E-LLM-3 与 Hopper/RA-L。

## 七件套（skill 规定输出）

1. **一句社区常识**："课程分数的价值在于分配；availability 即 utility；分数峰应随部署 N 走"——SFL/ADARFT/PLR 与本项目 v1 共享，其三条预言（oracle>stack、u16>u64、activity 可替 MaxMC）已被你们自己的数据全部反驳。
2. **一个有证据的真实瓶颈**：recycling 的 coverage 效应从未被原始测量（proxy-only；E2c 被 gate 关闭）；次瓶颈是 LLM 任务空间参数化（0.4 visits/prompt）；allocation 已饱和，从账本划掉。
3. **一个新的数学对象**：U_H(x;θ) ≈ A_N(p_x)·C(x;G,θ)（估计器精确 availability 项 × 池结构 compounding 项，ODE 为 C 的初版），以及策略耦合目的地测度 q_rec(·|θ_t) 及其漂移动力学。
4. **一个 identity-preserving baseline**：q_φ ∝ ρ·exp(βÛ)，β=0→uniform+floor；Û=log u_N→现 teacher；C≡const→现方法；gate≡1→上游 PLR；λ=0→纯 MaxRL。
5. **三个证伪实验**：branch-and-continue 预测子审计（+温度/峰位 2×2 分解）；RLOO+frontier teacher 塌缩测试；raw pass@k 的 recycling 重测。
6. **一个 future test matrix**：上表（regime × 预注册预测），reasoning-gym 与 Hopper 为未见配置，机器人行加质量门。
7. **一句 thesis**："Existing curricula optimize signal availability — the coefficient mass the deployed estimator can emit, exactly 2(pass@N − pass@1) — but training value is continuation utility. We reformulate task selection as availability-gated utility U ≈ A_N·C, realized as residual reweighting over a uniform-floor base that recovers it exactly at β=0, and test it where the factors disagree: structured pools, per-timestep signals under an availability gate, and hindsight recycling that rewrites the availability landscape itself."（中文版：现有课程优化的是估计器能发出的信号量，而训练价值是 continuation utility；我们把任务选择重写为 availability 门控的效用估计，在两者分歧的地方检验它。）

一句收尾：这次走查最重要的产出不是新实验清单，而是视角切换——你们的三个"负结果"放进新坐标后全变成了**正面证据**。这正是 skill 里"subsume, don't compete"在自己身上的用法。
