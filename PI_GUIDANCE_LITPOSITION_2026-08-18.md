# PI guidance — literature position, paper surgery, benchmark route (2026-08-18)

Saved verbatim. Received after the branching-pool v2 confirmatory result and the
paper polish commit 32aab36. Companion to `PI_JUDGMENT_2026-08-18.md` and
`SURPRISE_GUIDANCE_2026-08-18.md`.

---

你这次最新进度里，有一个特别值得肯定的地方：**你们不仅发现了新结果，还成功抵抗了"让结果看起来更漂亮"的诱惑。**

第一次 branching-pool 修正后，MaxRL 的 matched-activity transfer effect 是 (+0.00313)，看上去非常强；但你们没有把它直接写进论文，而是承认 matching criterion 是看过结果后才修正的，再用 fresh seeds 重做。重做以后，H=8 主检验虽然正向且 (p=.011)，但 effect (+0.00158) 没有跨过预注册的 (+0.002) SESOI，所以你们按规则给了 **inconclusive**；同时，H=20 的 frozen secondary 增长到 (+0.00724)，16/19 seeds 为正，却仍然没有把 secondary 偷换成 primary。

这并不是"结果变弱了"。恰恰相反，它让整个研究第一次出现了一个非常可信的新结构：

\[
\boxed{
\text{Estimator activity}
\neq
\text{Immediate utility}
\neq
\text{Long-horizon continuation utility}
}
\]

而且这个 separation 具有明显的 **horizon dependence** 和 **estimator dependence**。这已经足以成为下一篇论文的中心，但你们把它留在当前论文之外是正确的。最新修改只在当前稿里注明原 utility audit 是 (H=8)，加入 allocation 与 signal creation 的边界，并主动删掉对 branching result 的暗示，这个边界控制得很好。

下面我会先重新判断当前 paper 的真实位置，再做最邻近文献 review，最后给出具体的 paper surgery 和 benchmark 路线。

---

# 一、最新结果对当前论文意味着什么

## 1. 当前 paper 的核心不再是"我们找到了更好的 curriculum"

最稳、也最有辨识度的中心应该是：

> **A deployed group estimator induces its own finite-group task-activity geometry. For practical MaxRL, that geometry has an exact closed form. But activity describes where an update is available—not where training has the greatest long-horizon utility.**

中文就是：

> **一个 group estimator 会在任务空间上诱导出自己的有限样本 activity geometry。对 practical MaxRL，这个 geometry 可以精确推导；但它描述的是哪里存在可用更新，不是哪里具有最高的长期训练价值。**

这比"(u_N) 是更好的 teacher score"强得多，因为它能够容纳你们现在所有正面和负面证据：

* (u_{16}) 比 (p(1-p)) 更好：**estimator-conditioned activity shape 有经验内容**；
* exponent sweep 一直涨到 (u_{64})：**deployed-(N) peak location 不是正确 utility location**；
* Digits 中 universal mapping 失败：**activity shape 不是 universal sampler law**；
* AMaze 中纯 activity 替代 MaxMC 失败：**activity signal 的 bandwidth 可能不够**；
* branching pool 中 matched activity 后仍出现 H=20 transfer effect：**activity 不是 continuation utility**；
* RLOO 上这一 residual effect 小 40–55 倍：**continuation utility 的表现仍受 estimator 影响**。

你们最初的线性 chain audit 已经发现 (u_{16}) 在 MaxRL 下比 (p(1-p)) 更能排序真实 (U_H)，但它把 utility peak 放错了位置；随着 horizon 增长，rank correlation 没有崩掉，而 utility peak 向更低 (p) 的区域移动。之后又发现线性 chain 里 pass rate 和 downstream reach 几乎共线，根本无法干净检验 transfer。

所以当前 paper 最好的科学表述不是：

> "我们推导出了 optimal curriculum。"

而是：

> "我们推导出了 estimator 所允许的 activity，并测出了 activity 与 utility 开始分离的地方。"

这是一个很成熟的 theory-plus-boundaries paper。

---

## 2. 下一篇 paper 的中心已经浮现，但现在不要塞进来

你们 branching pool 的真正意义不是证明 (A_N C) 有效。事实上，三次测量都表明手工乘法几乎不提高 ranking：

\[
\rho(A_N,U_H)\approx \rho(A_N C,U_H).
\]

真正的发现是：

> 在固定相同 pass rate 和相同 coefficient activity 后，任务在 skill graph 中的位置仍然可以改变 long-horizon utility；而这种差异随 horizon 放大。

这直接支持你们之前讨论的 residual formulation：

\[
\widehat U_H(x)
=
A_{\mathcal E,N}(p_x)
+
\beta\,r_\phi(x,h,\mathcal G),
\]

其中 (r_\phi) 不重新学习 activity，而只预测结构、learner state 和 horizon 对 activity 的修正；(\beta=0) 时精确恢复当前 teacher。

但当前结果还没有授权一个 learned residual method：

* H=8 primary inconclusive；
* H=20 是 frozen secondary；
* branching substrate 是 synthetic；
* 尚未通过更复杂 stochastic learner 上的 identifiability gate。

所以你们现在的做法是对的：**当前 paper 只说 activity 不是 learning utility，不引入 continuation predictor；下一篇才把 H=20 和 residual utility 设成 primary。**

---

# 二、MaxRL 与最邻近文献：你们究竟站在哪里

我把现有文献按照"它优化或测量的对象"重新排列。这样比传统的 related-work 分类更能看出你们的独特性。

| 文献家族                              | 它研究的对象                                     | 它主要回答什么                                                  | 与你们最接近之处                                  | 你们不能声称什么                                                   |
| --------------------------------- | ------------------------------------------ | -------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| **MaxRL / RL2ML**                 | objective 与 finite-rollout update geometry | 给定 rollout budget，应使用什么 surrogate objective / estimator？ | estimator、group size、finite-sample update | 不能声称首次区分 population objective 与 finite-group update        |
| **ProCuRL / SFL / LILO**          | local learnability / ZPD                   | 当前哪些任务既非全会也非全不会？                                         | (p(1-p))、success variability              | 不能把所有 curriculum 都描述成 estimator-blind                      |
| **SEC / DUMP**                    | advantage magnitude / category bandit      | 哪些 category 当前产生较大 policy signal？                        | coefficient/advantage activity            | 不能说 advantage-based curriculum 都等于 (p(1-p))                |
| **PLR / ACCEL / PAIRED**          | regret、TD error、level replay               | 哪些环境值得重放、变异或生成？                                          | activity gating、frontier selection        | 不能把 one-bit success activity 当作 richer regret signal 的直接替代 |
| **TAC**                           | one-step cross-domain transferability      | 一次 task/domain update 是否帮助其他 domain？                     | activity 与 transfer 的 separation          | 不能宽泛声称首次发现 learnability 不等于 transfer                       |
| **SCRL / boundary-aware methods** | signal creation / external guidance        | all-fail dead zone 中怎样创造训练信号？                            | allocation 与 creation 的边界                 | 不能声称首次发现 hard samples 可无梯度                                 |
| **你们当前 paper**                    | task-level finite-group activity geometry  | 一个已部署 estimator 到底让哪些任务活跃？                               | exact coefficient-mass geometry           | 不需要证明 universal optimal sampler                            |
| **你们下一篇**                         | estimator-conditioned continuation utility | 哪个 training option 改善未来整段学习？                             | H-dependent transfer residual             | 必须超越 TAC 的 H=0/one-step transfer                           |

---

## 1. MaxRL：最重要的关系不是"基于它"，而是"审计它的 deployed convention"

MaxRL 的核心工作是把 RL 看成 maximum-likelihood objective 的低阶近似，通过 failure-event 的展开得到一族 compute-indexed truncated objectives；其理论中的 uncentered success-conditioned estimator，在 (N) 个 rollouts 下对应 truncation order (T=N)。MaxRL 还用 (w_T(p)) 描述不同 objective 对 pass rate 的 population-level reweighting。([arXiv][1])

你们最值得强调的不是"我们也使用了 MaxRL"，而是：

> **MaxRL 研究 objective-side weighting；我们研究 deployed finite-group coefficient vector 在 task space 中产生的 realized activity。**

而且你们的 (T=N-1) 结果必须非常明确地写成：

> 这不是修改或反驳 MaxRL 关于 uncentered estimator 的 theorem；它分析的是 released practical centered/control-variate convention，并包括 (K=0) 时把整组系数清零的行为。

这句话最好在 Lemma 后和 related work 中各出现一次。否则 reviewer 很容易误读成"作者声称 MaxRL 原论文 off-by-one"。

我建议直接使用这句英文：

> **This (N-1) correction does not revise MaxRL's theorem for its uncentered conditional estimator; it characterizes the released practical centered convention after its all-fail group is zeroed.**

---

## 2. RL2ML：这是目前最需要正面处理的最近邻

RL2ML 已经明确区分：

* population surrogate objective；
* finite-group stochastic update；
* group-level update scale；
* 不同 rollout budget 和 estimator parameterization 下的 metric/variance trade-off。

它还使用 Bernstein representation 描述 finite-rollout estimator family。([arXiv][2])

因此你们不要写成：

> "Prior work only studies population objectives; we are the first to analyze finite-group geometry."

这会被 RL2ML 直接击中。

更准确的 distinction 是：

> **RL2ML asks which finite-rollout objective and update scale to deploy while holding the data distribution fixed. We hold the deployed estimator fixed and ask which tasks its realized coefficient vector makes active for data selection.**

这句话非常重要。它把两个工作放在正交轴上：

\[
\underbrace{\text{Choose estimator/objective}}_{\text{MaxRL, RL2ML}}
\qquad\perp\qquad
\underbrace{\text{Choose task distribution given estimator}}_{\text{你们}}
\]

### 一个可以低成本加强理论的建议

在当前 paper 中显式定义任意 symmetric binary group estimator 的 activity geometry：

\[
\mathcal A_{\mathcal E,N}(p)
=
\mathbb E_{K\sim\mathrm{Binom}(N,p)}
\left[
\left\|\mathbf c_{\mathcal E}(K)\right\|_1
\right].
\]

若 estimator 的 coefficient vector 只依赖成功数 (K)，那么：

\[
\mathcal A_{\mathcal E,N}(p)
=
\sum_{k=0}^{N}
\binom Nk p^k(1-p)^{N-k}
a_{\mathcal E,k},
\qquad
a_{\mathcal E,k}
=
\|\mathbf c_{\mathcal E}(k)\|_1.
\]

换句话说，**每个 group estimator 都诱导一个 task-activity Bernstein polynomial**。MaxRL 的特殊价值是这个 polynomial 可以塌缩成：

\[
\mathcal A_{\text{MaxRL},N}(p)
=
2\{1-p-(1-p)^N\}.
\]

RLOO 又塌缩成 (2p(1-p))，GRPO 则保留 binomial sum。

这个定义可能数学上不复杂，但概念上很有用：它让论文从"发现了一个 MaxRL score"升级为：

> **提出 finite-group activity geometry 作为分析 data selection 与 estimator coupling 的对象，并为 practical MaxRL 得到 exact closed form。**

鉴于页面已满，我不会把它扩成新主 theorem；可以用一个 definition 加一行公式，并把一般 Bernstein 表达放进 appendix。

---

## 3. ProCuRL、SFL、LILO：你们应该 subsume canonical score，而不是 subsume 整个领域

ProCuRL 将 curriculum 与 zone of proximal development 联系起来；SFL 明确把 binary-success learnability 写为 (p(1-p))，并在 UED 环境中通过成功率寻找可学习 levels；LILO 同样围绕 success variance/learnability 选择 examples，并给出 expected-improvement 方面的理论与实验。([arXiv][3])

你们当前 TeX 中这句：

> "Every learnability curriculum in the literature scores tasks as if the learner were REINFORCE with one rollout."

范围太大，也容易被 SEC、DUMP、TAC 和 ProCuRL-Target 反驳。

建议替换为：

> **For binary-success tasks, the canonical learnability score (p(1-p)), used explicitly by SFL and LILO and recovered here as RLOO's coefficient mass, is the (N=2) slice of practical MaxRL activity.**

这仍然保留非常漂亮的 subsumption：

\[
u_2(p)=p(1-p),
\]

但不把所有使用 "learnability" 一词的方法都塞进同一个 algebra。

---

## 4. SEC 和 DUMP：它们说明 advantage activity 并非你们独有，但反而支持你们的 framing

SEC 用 category-level bandit 根据 mean absolute policy advantage 调整 sampling；DUMP 也把 difficulty/domain allocation 与 policy advantage、UCB-style exploration 结合。它们不是简单的 (p(1-p)) curriculum。([arXiv][4])

因此你们的 related work 可以这样写：

> Advantage-bandit curricula such as SEC and DUMP estimate activity empirically from observed policy advantages. We instead derive the exact expected coefficient activity induced by a specified binary group estimator before treating it as a curriculum hypothesis.

这里的差异是：

* SEC/DUMP：**在线观测一个 empirical activity proxy**；
* 你们：**从 estimator algebra 预先得到 exact task-level expectation**；
* 然后你们还测试了这个 exact activity 何时并不等于 utility。

这比"我们比它们更 principled"更准确，也更有说服力。

---

## 5. TAC：它不是当前 paper 的冲突，但会是下一篇最危险的 reviewer comparison

TAC 已经明确指出 local learnability 不足以描述跨领域收益，并将 mean absolute GRPO advantage 与 projected-gradient cosine transferability 结合；它的理论依据是一次更新后的 first-order Taylor improvement。实验使用六类 reasoning domain，并在 Qwen3-1.7B、Llama-3.2-3B 上测试。([arXiv][5])

所以当前 paper 不应声称：

> "We are the first to show activity or learnability differs from transfer."

但你们下一篇仍然有清楚的 wedge：

| TAC                                            | 你们下一篇                                         |
| ---------------------------------------------- | --------------------------------------------- |
| one-step / first-order transfer                | (H)-step continuation utility                 |
| domain-level gradient cosine                   | matched task-level causal continuation        |
| GRPO-centered                                  | estimator-conditioned MaxRL/RLOO dissociation |
| current projected gradients                    | learner-state- and horizon-dependent residual |
| assumes useful one-step signal is identifiable | first建立 CIR / oracle-precision gate           |
| H=0 transfer                                   | H=20 separation                               |

你们最新 H-sweep 正好是这条差异的雏形：H=4、8、20 的 matched-activity effect 单调变大，而 RLOO 同一效应小 40–55 倍。它支持"continuation—not merely transfer"的方向，但因为 H=20 目前只是 secondary，仍然应该留给下一篇重新预注册。

---

## 6. PLR、ACCEL、PAIRED：AMaze negative 是一个有价值的 signal-bandwidth boundary

PLR 使用 value/TD-error-based signals 估计哪些 levels 值得 replay；ACCEL 在高-regret levels 周围做变异；PAIRED 通过 adversarial regret 生成环境。它们的 signal 不是单次 binary success rate。([arXiv][6])

因此 AMaze 的结果应该始终写成：

> Activity did not lose because its theoretical shape was necessarily wrong; it lost because one Bernoulli observation per level visit was asked to replace a critic-derived signal observed at every timestep.

这解释了：

* pure replacement 失败；
* activity-gated MaxMC 恢复大部分 gap；
* 但 gating 仍未击败 upstream；
* PPO+GAE student 没有使用 MaxRL group estimator，因此 AMaze 不是你们 derivation 的 faithful mechanism test。

在你提供的最新快照中，full-budget AMaze confirmatory 已完成 15/20 training checkpoints，评估要等所有训练结束后才统一运行。

这个实验无论正负都不应改变 paper frame：

* 正面：activity shape 可以作为 richer replay signal 的 gate；
* 负面：进一步确认 signal bandwidth boundary；
* 两种情况都不能证明 (u_N) 是 standalone PLR priority。

---

## 7. Signal creation 文献：保留 allocation–creation 句子，但不要让 recycling 抢走 paper

SCRL、boundary-aware curriculum 和相近工作都在处理一个共同问题：对当前模型几乎不可能成功的问题，仅仅重新排序样本不能产生有效梯度，需要 subproblems、teacher guidance 或其他外部机制创造可验证信号。([arXiv][7])

所以你们新加入的这一句非常正确：

> A sampler redistributes existing signal; relabeling can create signal outside its support. At (A_N=0), no priority rule can help.

它给 recycling 一个清晰但次要的位置：

* activity geometry 描述 allocator 的可用信号；
* recycling 是超出 allocator ceiling 的 creation channel；
* 当前 paper 不需要再证明完整 hindsight method。

---

# 三、当前 paper 最应该做的七个修改

## 1. 保留当前标题，不要重新用 "The Estimator Decides"

当前标题：

> **Learnability, Reweighted: Which Tasks the Estimator Makes Active in Verifiable-Reward RL**

是准确的。

旧页面上的 **The Estimator Decides** 太强，因为最新结果已经证明 estimator 决定 activity geometry，但不能单独决定 continuation utility。

更准确的 public-site headline 可以是：

> **The Estimator Defines Activity**

或直接同步论文标题。

---

## 2. 把"一个 score"提升为"activity geometry"，但不要添加新方法

建议在 theory setup 前加入一句定义：

> Every finite-group estimator induces an activity geometry over task pass rates through the expected (\ell_1) mass of its realized coefficient vector.

然后再进入 MaxRL closed form。

这样 reviewer 看到的贡献顺序会变成：

1. 新分析对象：finite-group task activity；
2. practical MaxRL exact closed form；
3. 与 canonical learnability 的 exact factorization；
4. controlled evidence；
5. measured boundaries。

而不是：

1. 我们造了一个 score；
2. 这个 score 在一个 task 上赢了；
3. 但别处没赢。

两种叙事使用相同数据，研究分量完全不同。

---

## 3. 明确 MaxRL (T=N) 与 practical (T=N-1) 的 estimator distinction

这一点最好做成一行醒目的 interpretation，不要只埋在 appendix：

> **The shift from (N) to (N-1) is caused by the deployed centered-and-dropped convention, not by the maximum-likelihood expansion itself.**

这是很好的 audit finding，也避免被误判为 indexing error。

---

## 4. 收窄 related-work overclaim

把：

> Every learnability curriculum …

替换成：

> The canonical binary-success learnability score (p(1-p)) …

同时加入 RL2ML 和 TAC。下面这段可以直接作为 related-work 主干：

> **Objective-side work such as MaxRL and RL2ML characterizes which population or finite-rollout objective a group estimator implements, typically holding the task distribution fixed. Learnability curricula such as ProCuRL, SFL, and LILO adapt that distribution using success variability, while advantage-bandit methods such as SEC and DUMP estimate activity from observed policy advantages. TAC further augments local activity with first-order cross-domain transferability under GRPO. We study a complementary object: the exact task-level finite-group activity geometry induced by a deployed estimator, and empirically test where that geometry does—and does not—support curriculum selection.**

这段的优点是：它不与任何最近邻抢"first"，但非常清楚地说明为什么你们仍然独立。

---

## 5. 精简 abstract：现在有一点 benchmark salad

当前 abstract 同时承载：

* exact formula；
* factorization；
* Acrobot positive；
* cross-platform replication；
* exponent sweep；
* AMaze；
* Digits；
* maze factorial。

它们都是真的，但 reviewer 在一页内会遇到太多实验角色。

我的建议是 abstract 只保留四个 rung：

1. exact identity 与 factorization；
2. Acrobot controlled positive + replication；
3. peak-location falsification；
4. Digits/AMaze 作为两种 boundary，然后一句 methodological conclusion。

**maze factorial 从 abstract 移到 body。**

原因不是它不重要，而是它测试 estimator-conditioned coverage ordering，不是 activity score 本身。放在 abstract 里会模糊"这篇 paper 到底测的是 score、estimator 还是 curriculum safety"。

Abstract 最后一段可以压缩成：

> The controlled positive and two complementary failures support a calibrated conclusion: finite-group coefficient activity is a principled, estimator-conditioned source of curriculum hypotheses, not a universal measure of learning utility.

这会比列完所有 benchmark 后再解释 scope 更容易记住。

---

## 6. 做一张真正的 headline figure

现在最值得做的不是再加一张 training curve，而是一张 reviewer 看十秒就懂的三联图：

### Panel A：Identity

\[
u_N(p)=p(1-p)w_{N-1}(p)
\]

画出 (p(1-p))、(w_{N-1})、乘积和 peak trajectory。

### Panel B：What survives

* Acrobot (u_{16}-u_2) 的 frozen primary；
* 两个 cross-platform replication；
* Digits shape effects可以用浅色 supporting markers。

用 forest plot，不要放一堆 curve。

### Panel C：Where it stops

* exponent sweep：argmax 在 (u_{64})，而 deployed (N=16)；
* AMaze replacement negative；
* 可以用三个标签：

  * **shape supported**
  * **peak location rejected**
  * **standalone signal rejected**

这张图会让论文从"审计记录"变成"一张 identity 加一张 claim boundary map"。

---

## 7. 立即同步 public site、README 和 extended draft

这是现在最容易被忽略、但最现实的风险。

旧 public-facing notes 仍然出现：

* "Sampling by it is the curriculum"；
* "teacher and objective share the same compute knob"；
* "Zero difficulty hyperparameters"。

旧页面甚至仍把 (N) 说成同时决定 objective 和 optimal curriculum band。

这些句子已经被 exponent sweep 和 utility audit 超越。旧页面后面还把 "zero difficulty hyperparameters" 作为 benefit，而实际上 (\gamma)、uniform floor、posterior decay 和 tracking scheme 仍然是 empirical choices。

建议统一替换为：

* "Sampling by it is **a rollout-aware curriculum hypothesis**."
* "(N) determines estimator activity geometry; sampling concentration and tracking remain empirical choices."
* "The activity score has no hand-set target band, while posterior tracking, exploration floor, and sampling temperature remain algorithmic hyperparameters."

论文的诚实边界已经进步了，网站不能继续停留在更早、更兴奋的版本。reviewer 很可能先点 project page，而不是先读 appendix。

---

# 四、Dataset 与 benchmark：现在该押什么，不该押什么

评价 curriculum benchmark 不能只看"够不够大"。我建议使用六项 contract：

1. **Estimator alignment**：是否真的使用被分析的 group estimator；
2. **Frontier identifiability**：是否存在足够多 mixed-probability tasks；
3. **Revisitability**：teacher 是否能重复观察并更新 posterior；
4. **Transfer structure**：difficulty 和 downstream reach 能否被分离；
5. **Raw outcome retention**：能否重算 mean@k、pass@k 和 coefficient activity；
6. **Scale/external validity**：是否有 neural function approximation 或真实 reasoning task。

基于这个 contract，当前 benchmark 的位置非常清楚。

| Benchmark                    | 当前最适合回答的问题                                  | 优势                                                | 主要缺口                                       | 建议                           |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------- | ------------------------------------------ | ---------------------------- |
| **Acrobot fixed pool**       | rollout-aware shape 是否胜过 (p(1-p))           | frozen、paired、机制清楚                                | 小模型、结构较窄                                   | 保留为 controlled positive      |
| **Digits exact probability** | universal estimator-to-sampler mapping 是否成立 | exact (p)、几乎无测量噪声                                 | 外部规模弱                                      | 保留为 decisive counterexample  |
| **MAZE-SCORE**               | neural scale 下 (u_N) shape 是否仍有效            | MaxRL-aligned、procedural、raw outcomes、1.26M model | 单一 task family                             | **当前最高优先级**                  |
| **AMaze/minimax**            | one-bit activity 能否替代 critic replay signal  | 强 PLR/ACCEL baseline                              | PPO+GAE，不是 MaxRL mechanism                 | 只做 signal-bandwidth boundary |
| **GSM8K**                    | 真实 LLM RLVR data selection                  | verifier 真实、外部认可度高                                | 当前 7k prompts/低 revisits，posterior-starved | 当前 paper 不再追加                |
| **Countdown**                | exact relabeling / signal creation          | verifier 与 relabel interface 干净                   | 旧 pool 饱和、aggregate/raw-outcome 问题         | 当前 paper 不再追加                |
| **Reasoning Gym**            | procedural continuation curriculum          | 可生成、difficulty 可控、重复访问自然                          | 需先选不饱和 task families                       | 下一篇首选中型 benchmark            |
| **GURU multi-domain suite**  | cross-domain transfer 与 TAC 对比              | transfer 真实、六 domain                              | 成本高、TAC 已占据 H=0 baseline                   | 下一篇 external-scale benchmark |
| **Branching exact pool**     | matched activity 后 transfer 是否改变 (U_H)      | 因果隔离最强                                            | synthetic                                  | 下一篇第一 rung                   |

Reasoning Gym 提供大量 procedural generators、verifiers 和可控 complexity，因此特别适合需要反复测量 learner frontier 的 curriculum 研究。([神经信息处理系统大会论文集][8]) TAC 使用的 GURU 六-domain suite 则更适合最终验证跨领域 transfer，但它天然会要求你们与 TAC 的 projected-gradient H=0 baseline 正面对比。([arXiv][5])

---

## 为什么 MAZE-SCORE 是当前论文唯一值得继续押的规模实验

它同时满足：

* student 真正使用 MaxRL；
* score 的 (N) 与 deployed estimator 的 (N) 对齐；
* procedural task pool 有重复访问；
* raw binary outcomes 可以保留；
* neural model；
* 与 MaxRL 原论文的 procedural maze rung 直接连续。

MaxRL 本身的实验梯度从可控 exact comparison，到 procedural maze，再到 GSM8K 和更大数学 reasoning 模型；其中 procedural maze 是最适合隔离 estimator mechanism 的 neural rung，而 LLM benchmark 更适合验证最终 performance。([arXiv][1])

因此 MAZE-SCORE 不只是"再添一个大模型实验"，它是：

> **controlled Acrobot score-shape result与真实 RLVR LLM 之间最干净的 neural bridge。**

### 不论 MAZE-SCORE 最终正负，都应保留的 telemetry

在不改变 frozen primary 的前提下，descriptive mechanism figure 应至少保留：

* empirical task pass rate；
* posterior-estimated pass rate；
* predicted (u_N(p))；
* realized group coefficient mass；
* dead / mixed / all-success group fractions；
* selected-task pass-rate histogram；
* per-level raw binary evaluation outcomes；
* equal-update 与 equal-rollout accounting。

最有价值的 descriptive plot 是：

\[
\text{predicted }A_N(\hat p)
\quad\text{vs}\quad
\text{realized coefficient mass}.
\]

因为即使 curriculum performance 不显著，这张图仍能回答 theory 是否在 neural rollout process 中准确预测 **activity**。不要事后把它变成 performance primary，但它能把 negative 结果从"方法没赢"变成"activity calibration 成立，但 activity 未转化为 utility"。

---

## 为什么现在不应该再追加 GSM8K 或 Countdown

你们自己的旧实验记录已经给出足够明确的诊断。

GSM8K 的 teacher 面对 7,473 个 prompts，3,200 次 group draws 平均每个 prompt 不到一次访问；posterior 虽然学到某些 difficulty 信息，但 sampling policy 几乎没有足够 revisits 去采取行动，属于 posterior starvation。

Countdown 的第一版 pool 接近 random-guesser ceiling，导致 allocation 和 recycling 都没有足够空间；后来的 aggregate 又存在 mean@16 上升、logged bootstrap proxy 下降、缺失完整 raw task outcomes 的问题。

再增加 seed 并不能修复 benchmark contract。那只会让 paper 多一个分支，而不是多一个结论。

---

# 五、MAZE-SCORE 与 AMaze 的结果分支应该提前写死

## MAZE-SCORE

### 若 `u_N > p(1-p)` 获得支持

论文主张升级为：

> rollout-aware activity shape survives from exact/controlled environments to a neural procedural learner.

但仍然不能说：

* deployed-(N) peak optimal；
* activity equals utility；
* beats PLR；
* universal curriculum method。

### 若 inconclusive

如实写：

> the neural estimate is directionally X with interval Y; the study does not establish an effect at or above the registered SESOI.

不要追加 seeds，不要换 endpoint。paper 仍然拥有 exact theory、replicated controlled positive 和 preregistered boundaries。

### 若 practically ruled out

论文会变成更锋利的 scale-boundary paper：

> exact estimator activity predicts controlled task selection but does not automatically transfer to neural procedural learning at the registered effect size.

这不是最理想的 submission shape，但比 post-hoc tuning 更可信。此时也不要把 branching H=20 result搬进来救场，因为那属于另一个 research object。

---

## AMaze

### 若 activity-gated MaxMC 击败 upstream

只能说：

> estimator-inspired activity gating can improve a richer regret signal.

不能说 (u_N) 本身成为 competitive PLR priority，因为：

* gate 保留了 MaxMC signal；
* student 是 PPO+GAE；
* mechanism 与 MaxRL derivation 不一致。

### 若未击败

直接关闭该 lane：

> activity is useful as a diagnostic shape, but the one-bit level statistic lacks the bandwidth to replace or improve a tuned regret replay system under this protocol.

不要再调 decay、(N)、gate form 或 per-maze exceptions。

---

# 六、下一篇 continuation paper 的正确实验顺序

当前 paper 结束后，不应直接训练 residual predictor。正确顺序仍然是你们已经接受的：

## 第一关：G2 identifiability gate

在 stochastic Acrobot 或其他 neural learner 上，先测：

\[
\mathrm{CIR}_{H,L,R}
=
\frac{\operatorname{Var}_x[\mathbb E U_{H,L}(x)]}
{R^{-1}\mathbb E_x[\operatorname{Var} U_{H,L}(x)]}.
\]

只有 oracle continuation values 在选定 (H,L) 下可稳定排序，才有资格比较 learned scheduler。

这可以避免再次出现"oracle label 自己都不稳定，却训练一个更复杂 predictor"的问题。

## 第二关：重新以 H=20 作为 primary

下一次 branching study 应该：

* fresh seeds；
* H=20 primary；
* H=4/H=8 作为 horizon curve；
* 同时匹配 (|\Delta p|) 与绝对 (|\Delta u_N|)；
* 固定 transfer mismatch threshold；
* H=20 的 SESOI 在数据前冻结；
* MaxRL 与 RLOO 都保留；
* (A_N C) 只作为已知 negative control，不再作为 proposed model。

## 第三关：与 TAC 的 H=0 baseline 正面对比

下一篇的方法表中至少要有：

* activity only；
* exact immediate gain；
* TAC-style projected gradient alignment / H=0 transfer；
* exact H-step continuation oracle；
* learned residual continuation score。

最关键的 contrast 是：

\[
\text{H=20 residual}
-
\text{H=0 transfer alignment}.
\]

否则 reviewer 会认为你们只是把 TAC 换了 estimator。

## 第四关：从 branching pool 到 Reasoning Gym，再到 GURU

最合理的 escalating ladder 是：

1. **Exact branching graph**：证明 matched activity 后 continuation structure 存在；
2. **Reasoning Gym subset**：procedural、可重复、可控 difficulty；
3. **GURU multi-domain RLVR**：真实跨 domain transfer，与 TAC 比较；
4. 只有前三层成立后，才进入更昂贵的大模型 benchmark。

---

# 七、现在应当停止的工作

当前 paper 不再做：

* 新 (u_N) exponent tuning；
* 新 GSM8K seeds；
* 新 Countdown rescue；
* 新 continuation predictor；
* 为 AMaze 持续调 gate；
* 将 H=20 secondary 升格或暗示成当前 paper evidence；
* 把 recycling重新变成主贡献。

当前只完成三件事：

1. 冻结完成 AMaze analyzer，并按既有规则报告；
2. 完成 MAZE-SCORE，不改协议；
3. 做 manuscript、site、README 的 literature/claim synchronization。

---

# 最终判断

这篇 paper 现在最有价值的，不是证明"MaxRL 搭配我们的 curriculum 更强"，而是建立下面这条以前被混在一起的层级：

\[
\boxed{
\text{Population objective weighting}
\rightarrow
\text{finite-group coefficient activity}
\rightarrow
\text{immediate transfer}
\rightarrow
\text{continuation utility}
\rightarrow
\text{signal creation}
}
\]

MaxRL 和 RL2ML主要研究第一层；SFL、LILO、SEC、DUMP主要处理第二层附近；TAC走到第三层；你们当前 paper 给第二层一个 exact estimator-conditioned definition，并测出它与后面几层的边界；你们下一篇则有机会真正进入第四层。

因此我会坚持当前标题，并把 abstract-level thesis 收紧成这一句：

> **A group estimator induces a finite-group activity geometry over tasks. For practical MaxRL this geometry is exactly (2(\mathrm{pass@}N-\mathrm{pass@}1))，or learnability reweighted by MaxRL's own objective weight. It predicts useful task-selection structure in controlled settings, but its peak, signal resolution, and long-horizon utility are distinct empirical questions.**

这句话既保留了你们最漂亮的 identity，也容纳了你们最诚实、最有价值的 negative results。它不会因为 MAZE-SCORE 或 AMaze 的单个结果翻转而倒塌——这正是现在这个 paper 比几天前更强的地方。

[1]: https://arxiv.org/html/2602.02710 "https://arxiv.org/html/2602.02710"
[2]: https://arxiv.org/abs/2605.30154 "https://arxiv.org/abs/2605.30154"
[3]: https://arxiv.org/abs/2304.12877 "https://arxiv.org/abs/2304.12877"
[4]: https://arxiv.org/html/2505.14970 "https://arxiv.org/html/2505.14970"
[5]: https://arxiv.org/html/2606.25178v2 "https://arxiv.org/html/2606.25178v2"
[6]: https://arxiv.org/abs/2010.03934 "https://arxiv.org/abs/2010.03934"
[7]: https://arxiv.org/abs/2605.22074 "https://arxiv.org/abs/2605.22074"
[8]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/528d56195a2c77c808494c86fa7c77ad-Abstract-Datasets_and_Benchmarks_Track.html
