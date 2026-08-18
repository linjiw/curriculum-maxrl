# PI judgment — 2026-08-18: from active tasks to useful tasks

*Received from the PI on 2026-08-18, after the utility audit result. Saved
verbatim. The working plan derived from it follows the guidance.*

---

我先给一个不绕弯的判断：

**CurriculumMaxRL 现在已经不应再被定义成"从 MaxRL 推导出一个更好的 curriculum score"。**
它真正形成的研究主线是：

> **Estimator 决定一个任务是否会产生可用更新，但 estimator activity 并不等于这个任务对未来学习有价值。**

这不是退一步，而是一次很关键的坐标升级。你们最值得认可的进步，不只是 Acrobot 上又赢了多少，而是面对 peak-location、AMaze、Digits、Countdown 等反例时，真的砍掉了原来最诱人的强主张。现在留下来的 exact identity 很稳：MaxRL 的任务级 coefficient activity 是 A_N(p)=2(pass@N-pass@1)，但"这个 activity 的峰值就是最优 curriculum 应该采样的位置"已经被实验否定。

换句话说，项目并没有散掉。它是在逐渐显露一个更清楚的三层结构。

# 一、目前 CurriculumMaxRL 实际解决了什么

## 第一层：Estimator support / activity——已经相当扎实

你们现在可以准确回答：对于一个成功率为 p 的任务，在有限 rollout group 下，MaxRL 到底有多大概率产生非零、非平凡的更新？答案不是传统的 difficulty，也不是泛化的 learning progress，而是 A_N(p)。它描述了一个任务处于：全失败、没有更新的 dead zone；至少偶尔成功、同时仍有 headroom 的 active zone；几乎全部成功、优势系数趋于消失的 mastered zone。

这个结果真正重要之处，不只是给出了一个采样公式，而是证明了：**"learnability"不是任务本身固定的属性，它取决于底层 estimator。** p(1-p) 对 RLOO 是自然的 activity quantity；MaxRL 则通过自己的 likelihood-style weight function 对它重新加权。

所以当前论文最硬的理论对象不是"最优 curriculum"，而是：**estimator-induced task activity geometry。** 这部分可以保留，而且应该成为当前论文的中心。

## 第二层：Allocation 与 signal creation——你们已经把两件常被混在一起的事拆开了

teacher 只能重新分配已有信号，避免把 rollout 花在 dead 或 mastered task 上；hindsight recycling 能把原本全失败的轨迹变成其他目标下的 verified success，从而创造原先不存在的信号；objective 决定这些 data intervention 是否安全。你们的 oracle 对照进一步说明，pass-rate posterior 已经接近分配上限；真正还能超越 perfect allocation 的是 recycling，而不是更精确地估计 p。

哪里有信号？→ A_N(p)。哪里没有信号但可以制造信号？→ recycling。这两个问题你们已经回答了相当一部分。

## 第三层：Training utility / continuation value——这是现在真正没解决的核心

你们尚未回答的是：在所有"有更新"的任务中，训练哪一个会让未来 deployment objective 提升最多？这是 activity 无法单独决定的。两个任务可以拥有完全相同的 p、相同的 A_N(p)、相同的 rollout cost，甚至相同的当前梯度范数，但它们的长期价值可能完全不同。

因此你们现在最清楚的研究层级应当写成：

**estimator activity ≠ immediate target gain ≠ continuation utility**

# 二、按照 Kaiming-style SURPRISE，重新诊断这个项目

## S — 应该攻击的社区常识

> **"如果一个任务让当前 estimator 产生更大的 coefficient activity，那么多训练这个任务通常会带来更大的未来性能收益。"**

它隐含了一个未经证明的跳跃：update exists ⇒ update is useful。而你们自己的结果已经显示这个跳跃不成立。

## U — 真正的 bottleneck 不是 pass-rate estimation

1. p 已经不是主要瓶颈：matched oracle 与便宜 posterior teacher 几乎打平。
2. activity curve 的精确峰值也不是瓶颈：部署 N=16 时最优点跑到 u_64；再扫 u_32/u_48/u_96 只是继续调 surrogate。
3. 真正瓶颈是 transfer、persistence 与 signal bandwidth：AMaze 说明 A_N 更像可用性门控器，不是完整的 utility。
4. 评估目标本身仍存在错位：Countdown proxy、GSM8K 访问预算不足。

当前 bottleneck ledger 的第一名：**缺少从一次 task-conditioned update 到未来 quality-qualified deployment performance 的因果价值估计。**

# 三、下一步最值得押注的研究方向

**From Active Tasks to Useful Tasks: Residual Continuation-Value Curricula for Verifiable-Reward RL**

> Existing curricula prioritize tasks by current estimator activity, but deployment performance depends on how task-conditioned updates transfer and persist through future optimization. We estimate only the residual continuation value beyond an activity-preserving baseline and test it under unseen task structures, policy stages, and estimator configurations.

当前论文：estimator 决定哪些任务 active；下一篇论文：在 active tasks 中，什么决定哪些任务 useful。

# 四、新的数学对象：quality-qualified continuation utility

U_H^Q(x;θ) = J_Q(Train_H(θ; x, π_c)) − J_Q(θ)

第一个 update 使用任务 x；后续 H−1 个 update 使用共同的 continuation schedule π_c；所有候选任务共享相同 continuation randomness，形成 paired causal contrast；J_Q 是 quality-qualified deployment objective：max Δmean@1 s.t. Δpass@k ≥ −ε, Δcoverage ≥ −ε_c, Δcompute ≤ 0。

# 五、保留当前方法作为 identity path

ρ_A(x) = (1−ε)(A_N(p̂_x)+δ)^γ / Σ(...)^γ + ε ρ_0(x)
R_H(x;θ) = U_H^Q(x;θ) − b(A_N(p̂_x), c_x, s_θ)
q_{β,φ}(x) ∝ ρ_A(x) exp(β R̂_{φ,H}(x)),   β=0 ⇒ q = ρ_A

# 六、第一版 predictor 不要用 Transformer 或 world model

只使用已经付过 rollout 成本的 charged trajectory data：p̂_x、K/N 与 coefficient mass；success/failure-conditioned gradient sketch；gradient variance 与 update norm；rollout length、terminal reason、achieved depth；task context 与 policy stage；后续自然训练中观测到的 update persistence；rollout cost。固定 random projection + ridge / 低秩 bilinear。第一版如果连线性、低秩模型都无法在 held-out continuation oracle 上产生稳定 ranking，就不要用更复杂网络掩盖失败。

# 七、三个会真正推翻主张的实验

**实验一：Activity-matched、transfer-mismatched task pairs。** 相同 p、A_N(p)、rollout cost、immediate gradient norm，但共享技能结构不同。测 H=5,20,100 的 branch-and-continue utility。推翻条件：严格 activity matching 后两类任务的 continuation utility 没有稳定差异，或 A_N 的 ranking 已经和 oracle 一样好。

**实验二：Horizon-dependent ranking reversal。** 比较 A_N(p), U_0, U_5, U_20, U_100。共同预生成的 continuation schedules；oracle label 在多个独立 schedules 上平均。推翻条件：ranking 随 horizon 基本不变，或 one-step gain 在所有 H 下始终最佳。

**实验三：Deployable G2 gate。** 相同初始化 + uniform burn-in；只用 charged trajectory data。arms：uniform / activity teacher / +residual predictor / +shuffled residual / β=0 identity / branch-and-continue oracle ceiling。equal interaction/updates/eval；报 raw per-seed mean@k、pass@k、AUC、coverage、cost、方差。推翻条件：residual predictor 无法在 held-out seed 与 held-out task graph 上预测 oracle ranking；shuffled = learned；online scheduler 不优于 activity baseline；提升只在 speed 而 final/coverage 恶化。**在这个 gate 通过之前，不进入 Acrobot confirmatory campaign。**

# 八、future test matrix

| 阶段 | 关键变化 | 真正测试的问题 | 通过后才能进入 |
|---|---|---|---|
| Exact synthetic | independent、chain、branching task graph | activity 与 continuation utility 是否可分离 | deployable predictor |
| Exact/CPU control | policy stage、H、skill sharing | predictor 是否跨训练阶段稳定 | Acrobot |
| Neural maze | goal-distance transfer vs maze-size cliff | 是否识别"可迁移的 curriculum axis" | 更大 neural benchmark |
| Estimator shift | MaxRL N=8/16/32、RLOO | residual utility 是否超越 estimator-specific activity | estimator-general claim |
| Unseen composition | 新 task graph、新初始能力 | 预测未来配置而非记住任务 ID | robotics |
| Humanoid | unseen motion family × dynamics | tracking 提升且不抖动/滑步/耗能 | 真机 |

# 九、当前 ICLR 稿件应该如何收口

> **Task curricula are not estimator-agnostic. Each group estimator induces a task-activity geometry, but activity is only a diagnostic of available update—not a universal measure of learning utility.**

1. 删除"sampling by it is the curriculum"的强表达 → "a principled baseline or gate when task activity is the binding bottleneck."
2. 修改"compute knob and curriculum knob are the same knob" → "N determines which tasks the estimator makes active, but not necessarily which active tasks maximize learning utility."
3. "zero difficulty hyperparameters" → "no manually specified ZPD center or width is required to define the estimator-activity score." sampler 仍有 γ 与 floor。
4. Hindsight 保留为 complementary channel：allocation redistributes existing signal; recycling can create signal outside the allocator's support。不在同一篇 9 页论文里同时承诺 universal sampler、recycling framework、GRPO safety law、LLM scaling 和 continuation utility。
5. MAZE-SCORE 是当前论文的 closing experiment，不是下一篇方法。不要趁这个实验加入 continuation predictor。

# 十、Kaiming-style 最终研究判断

一句社区常识：Tasks that induce more estimator coefficient activity are more valuable curriculum choices.
一个真实瓶颈：估计 task-conditioned update 在未来优化中的 transfer 与 persistence。
一个新的数学对象：U_H^Q(x;θ) = J_Q(Train_H(θ;x)) − J_Q(θ)。
一个 identity-preserving baseline：q_{β,φ} ∝ ρ_A exp(β R̂), β=0 ⇒ q=ρ_A。
三个推翻实验：activity-matched transfer-mismatched pair；H=0/5/20/100 ranking reversal；trajectory-only deployable residual scheduler with identity/shuffle/uniform/oracle。
一个 future test matrix：independent → chain → branching；early → mid → late；N shift → estimator shift；exact → neural → control → humanoid；all with mean@k, pass@k, coverage, cost, quality.
一句 thesis：Existing curricula optimize where the estimator is active, but future deployment performance depends on whether task-conditioned updates transfer and persist through subsequent learning. We estimate residual continuation value on top of an activity-preserving sampler and evaluate it under unseen task structures, policy stages, and estimator configurations.

最直接的下一轮工作因此不是继续找更好的 u_N，而是：**冻结当前 activity paper 的边界，完成 MAZE-SCORE；同时在 exact task graph 上实现 shared-continuation branch oracle 与 trajectory-only G2 residual predictor。只有 G2 通过，才解锁 Acrobot 和更大规模实验。**
