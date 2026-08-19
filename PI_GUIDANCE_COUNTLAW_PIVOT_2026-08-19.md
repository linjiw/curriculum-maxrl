# PI guidance — pivot to task granularity and count-law activity (2026-08-19)

Saved verbatim. Received after `262b1ea`. Read alongside
`EDITORIAL_CHARTER_2026-08-19.md`; where the two conflict, the conflicts are
enumerated and adjudicated in `GUIDANCE_RECONCILIATION_2026-08-19.md`.

## Verification performed before adopting any of it

The central new proposal — a common count-law framework
`A_E(z) = Σ_k P(K=k|z)·M_E(k)` with per-estimator realized masses — was checked
against the **deployed** estimators in `curriculum_maxrl/estimators.py` over
N ∈ {4,8,16,32} and every k:

| estimator | proposed `M_E(k)` | agreement with deployed code |
|---|---|---|
| practical MaxRL | `2(1−k/N)·1{k>0}` | exact (< 1e-9) |
| RLOO | `2k(N−k)/(N(N−1))` | exact (< 1e-9) |
| GRPO, sample-SD | `(2/N)·√((N−1)/N)·√(k(N−k))` | matches the zero-ε idealization; deployed differs by exactly `× std/(std+EPS)` with `EPS = 1e-6`, verified to 1e-12 |

All three are correct. The GRPO deviation is the finite-ε denominator
stabilizer the manuscript already scopes, not an error in the formula.

---

[Guidance text as received follows.]

# 我的总体判断

我认真读完了你 8 月 19 日的最新研究记录和 extended draft。我的真实判断是：

**这项工作正在从一篇证据很多、但主线分散的"curriculum system paper"，转变成一篇更有机会打动 ICLR reviewer 的科学发现型论文。** 真正让它变强的，不是又多了一个正结果，而是 MAZE-SCORE 的负结果迫使你识别出了原来没有被说清楚的隐藏变量：**curriculum 所评分的 task unit，是否与 estimator 实际消费的随机单元一致。**

你没有把负结果藏起来，而是继续追问"为什么在 Acrobot、Digits 上成立，在 neural maze 上反转"，最后得到任意 joint group law 下的恒等式。这不是退让，而是论文发生了真正的认识升级。当前最重要的结果已经不是 u_N(p) 是比 p(1-p) 更好的 curriculum score，而是：

> **有限 group estimator 的 activity 是 joint rollout law 的函数；mean pass rate 只有在 curriculum 的 scored unit 与 estimator 的 atomic unit 对齐、且条件 i.i.d. 时，才足以描述这种 activity。**

A_N(Q) = 2(Pr_Q(K>0) − E_Q[K]/N) 不需要 independence，也不需要 identical distribution；而把一个 coarse unit 的平均通过率 p̄_z 塞进 i.i.d. 公式，会产生一个精确的 aggregation gap
A_N(p̄_z) − A_N(Q_z) = 2[Pr(K=0|z) − (1−p̄_z)^N]。
这正是目前整篇论文里最值得押注的结果。

# 一、从 ICLR reviewer 视角，现在这篇论文处在什么位置

| 版本 | 当前可能评价 | 核心原因 |
| --- | --- | --- |
| Extended research draft 原样投稿 | Weak Reject，约 4/10 | 理论、teacher、hindsight、GRPO coverage、gate、Countdown、GSM8K、Jugs、PLR 同时争夺主线 |
| 目前 compact draft 的新 pivot | Borderline，约 5/10 | 新观点明显变强，但 granularity 解释仍主要来自跨实验的 post-hoc 归纳 |
| 加入 prospective granularity intervention 并大幅裁剪 | 有机会到 Weak Accept，约 6/10 | 一个清楚的新问题、一个精确答案、一个前瞻性因果验证、一个诚实边界 |

现在的问题在于：**你已经有了新知识，但 PDF 仍像在努力证明 FrontierMax、hindsight 和 MaxRL coverage 是一个统一且普遍有效的系统。**

# 二、真正应该成为论文开头的反例

两个 curriculum level，平均 pass rate 都是 0.5。

- **Level A**：每个 task 都以 p=0.5 成功。group 里通常既有成功也有失败，estimator 产生很强 contrast。
- **Level B**：一半 task p=1，一半 p=0。平均 pass rate 同样 0.5，但每个 group 要么全成功（无 contrast）要么全失败（被丢弃），真实 coefficient activity 是零。

任何只看 level mean pass rate 的 f(p̄) 方法都认为这两个 level 一样，但 estimator 看到的是两个相反的世界。问题不再是"应该用 p(1−p)、u_16 还是 u_64"，而是：**这个 p 究竟属于什么随机单元？它是否足以决定 group outcome law？**

# 三、我建议冻结下来的核心 thesis

推荐标题：**When Pass Rate Is Not Enough: Task Granularity and Estimator Activity in Verifiable-Reward RL**

一句话 thesis：
> Finite-group estimators act on a joint rollout law, so mean pass rate is a sufficient curriculum statistic only when the curriculum scores the same atomic unit that the estimator consumes; at coarser units, activity must be estimated from the group-count law, and even exact activity is not learning utility.

三层断裂：p̄_z ⇏ P(K|z) ⇒ A_E(z) ⇏ ΔJ ⇏ long-horizon curriculum value。

# 四、理论上最值得增加的一步：从 MaxRL identity 推广到 group-count law framework

对 estimator E，定义 M_E(k) = Σ|w_i| given K=k。只要 estimator permutation-equivariant，则对任意 curriculum unit z：
A_E(z) = Σ_k P(K=k|z)·M_E(k)。
**Estimator activity 的充分统计量不是平均 pass rate，而是 success-count law P(K|z)。**

| Estimator | M_E(k) |
| --- | --- |
| Practical MaxRL | 2(1−k/N)·1{k>0} |
| RLOO | 2k(N−k)/(N(N−1)) |
| GRPO, sample-SD | (2/N)√((N−1)/N)·√(k(N−k)) |

## 为什么使用 L1 coefficient mass，也必须补一个解释

1. 对 live MaxRL group，coefficients 和为零，所以 ½‖w‖₁ 同时等于总 positive 和总 negative coefficient mass——它测量 estimator 能形成多少 success-versus-failure contrast。
2. 若 ‖S_i‖ ≤ B，则 ‖Σ w_i S_i‖ ≤ B‖w‖₁。coefficient mass 是只依赖 estimator、在未知 score geometry 下的 worst-case update envelope。
3. 它不是实际 gradient norm，也不是 expected improvement；真实效果还取决于 μ₊−μ₋ 的方向。

# 五、把现有 teacher 改造成真正由新理论导出的方法：Group-Law Activity Teacher

对每个 unit z 维护 group success count 的分布 π_z(k)=P(K=k|z)（带 decay 的 Dirichlet），按部署 estimator 计算 Ã_E(z)=Σ_k π̃_z(k)M_E(k)。对 practical MaxRL 简化为 q̃_z − p̄̃_z。性质：atomic conditional-i.i.d. 时退化回 u_N(p)；heterogeneous level 上不必假装单一 Bernoulli；自然处理 correlated / anti-correlated / non-identical groups；同一 posterior 服务三种 estimator，只需换 M_E(k)。K 本来就会被观测到，没有额外 rollout 成本。

# 六、下一步最关键的三个实验

1. **Controlled coarsening experiment（非做不可）**：同一 substrate、相同 atomic tasks / warmstart / estimator / budget，只改变 curriculum 如何把 tasks 聚成 units；partition width b ∈ {1,2,4,8,…}。四个 arms：uniform；naive plug-in u_N(p̄̂_z)；group-law score Σ_k π̂_z(k)M(k)；atomic oracle E_{x|z}[u_N(p_x)]。关键控制是构造 **same-mean, different-heterogeneity** matched aggregates。Primary 先看 activity calibration error |Â(z) − A_realized(z)|，再把 held-out AUC 作为 downstream primary 或 co-primary——这允许两种都有科学意义的结果。paired warmstarts，≥20 paired seeds，运行前冻结 endpoint / SESOI / partition construction / 判定规则。
2. **Neural MAZE prospective correction（最高价值 scale experiment）**：不要再比较 u_16/u_32/u_64；比较 p(1−p)、naive u_32(p̄̂_level)、group-law q̂_level−p̄̂_level，数据允许时再加 finer task posterior。相同 warmstart / seed block / rollout groups / budget，level 内 task sampling 固定。用现有 48-block variance 做正式 power calculation；预设 treatment-delivery gate（若两个 sampler 分布几乎相同，endpoint 不能当作 granularity mechanism 的检验）。
3. **LLM experiment：只做一个能真正 deliver treatment 的版本**：procedural reasoning pool；8–20 个反复访问的 family × difficulty cells；前 10–20% budget uniform calibration；比较 uniform / mean-p plug-in / group-law；保留 raw outcomes 计算标准 mean@k 与 pass@k；matched generation tokens；≥3 seeds；预注册 sampler TV distance、cell ESS、rank divergence 作为 delivery gate。训练前先用 frozen checkpoint 做便宜的 group-law calibration test。

# 七、哪些内容应该从 ICLR 主文中拿掉

1. **Hindsight/recycling 不应再与主论文平起平坐。** 最准确的表述不是"hindsight gradients are exact"：relabeling 通常是 adaptively selected、off-policy 的 auxiliary update；当前 LLM 实现还会在一个 group 中混合多个 achieved destinations，共享 K 会耦合不相关 tasks。加上 Countdown 缺 raw outcomes、extra-update dose 可解释大部分收益、corrected gate 未验证、Jugs 显示 recycling 会加速 pool-conditional collapse——主文只保留一句 conceptual implication，其余移到 appendix 或下一篇。
2. **"Estimator decides safety" 必须降级** 为 "estimator conditions the activity geometry and can alter the coverage–reliability trade-off under a curriculum"。
3. **AMaze、IsaacLab 和 Gym demos 不应继续占主文**，它们适合 boundary appendix。

# 八、现在必须清理的事实和表述矛盾

- **MAZE-SCORE 状态矛盾**：文末仍写它是 highest-value pending experiment，必须全局删除旧状态。
- **Oracle 的解释错误**：posterior teacher 0.728、oracle 0.8885、full stack 0.8895。正确解释是 recycling 补偿了 posterior 的 allocation error，而不是 posterior 已接近 oracle。
- **"The corollary predicts the sign of every score contrast" 过强**：它能预测的是 u_N(p̄_z) 对真实 activity 的 calibration bias 非负，不能直接预测 downstream AUC 的符号。现有 sign table 只能称 "consistent with the granularity diagnosis"。
- **"Aggregation penalty grows with N" 也过强**：对固定 heterogeneous distribution，E[(1−p_X)^N] − (1−E[p_X])^N 不保证对所有 N 单调增加；当所有 p_X>0 时两项最终都趋近零。安全的说法是 larger N can amplify sensitivity to heterogeneity around the active region。
- **Cross-estimator magnitude 不要作为 headline**：global scale 可被 learning rate 吸收；应强调 normalized shape、exact zeros、tail concentration、matched or swept learning rates。

# 九、建议的 9 页结构

1. Introduction 1 页（same mean、different group law 的反例；三项贡献）
2. Group estimators and task units 0.75 页
3. Count-law activity theory 2 页
4. Group-Law Teacher 0.75 页
5. Controlled coarsening 2 页
6. Neural-scale test 1.5 页
7. Boundary / related work / limitations / conclusion 1 页

最多四张主图：same p̄ different P(K)；theory + aggregation gap；controlled coarsening；neural maze correction。

# 十、目标 abstract（方括号处实验完成后才可填）

> Curriculum methods for RLVR commonly summarize each task unit by a pass rate p and apply a learnability curve f(p). This assumes the unit scored by the curriculum is the same random object consumed by the group estimator. It need not be: a unit containing uniformly uncertain tasks and one mixing mastered with impossible tasks can have the same mean pass rate, while only the former produces mixed-outcome groups and nonzero updates. … A_N(Q) = 2(Pr_Q[K>0] − E_Q[K]/N). The familiar 2(pass@N − pass@1) is only its conditionally-i.i.d. atomic slice. For a coarse unit z, plugging its mean pass rate into this curve overestimates true activity by exactly twice its excess all-fail probability. We therefore estimate the group success-count law directly … In controlled matched-mean and neural maze studies, [prospective result]. These findings distinguish update availability from learning utility.

# 十一、从现在到投稿的执行顺序

- 8/19–23 冻结新 thesis；完成 count-law theorem、L1 interpretation、Group-Law Teacher；清理 stale claims。
- 8/23–30 预注册并完成 controlled coarsening（submission 的最低必要实验）。
- 8/27–9/7 neural MAZE prospective correction；先确认 treatment delivery。
- 9/1–10 只在 frozen calibration 显示足够 heterogeneity 时才做 LLM。
- 9/8–15 重写 9 页正文，两位局外人做 adversarial mock review，只回答：一句话 claim 是什么、哪个实验直接检验它、哪个结果会推翻它。
- 9/16–18 冻结 abstract、作者、title。
- 9/19–24 只做 correctness / anonymity / figure audit。

# 最终研究决策

> **冻结 universal curriculum claim；将论文 pivot 到 task granularity 与 group-law activity；实现 count-law correction；用一个 controlled coarsening experiment 和一个 prospective neural experiment 验证它；把 hindsight、gate、coverage safety 和其余 domain ladder 移出主线。**

**同一个平均通过率，并不意味着 estimator 看见了同样的学习机会。**
