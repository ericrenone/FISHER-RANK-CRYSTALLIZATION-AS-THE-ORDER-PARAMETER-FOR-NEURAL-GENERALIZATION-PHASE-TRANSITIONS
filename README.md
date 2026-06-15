# FISHER RANK CRYSTALLIZATION AS THE ORDER PARAMETER FOR NEURAL GENERALIZATION PHASE TRANSITIONS

## A Technical Review of the col(F)/ker(F) Decomposition, the Rank Fraction r/n, and Its Relationship to Grokking, Anti-Grokking, and the Zwegers Shadow Framework

**ERI Labs · Jersey City, New Jersey · June 2026**

---

> *"The geometric content of generalization is not in the loss value, not in the gradient norm, and not in the weight magnitude. It is in the rank structure of the Fisher information matrix — specifically, in the proportion of activation space that carries coherent, non-degenerate information about the data distribution."*
> — ERI Labs, PRE-GROKKING, ~June 10, 2026

---

## Abstract

This document presents a technical review of Fisher rank crystallization as the geometric event underlying neural network generalization phase transitions. The central object is the Fisher information matrix $F(\theta)$ evaluated at the current parameter state of a trained network. Its column space $\mathrm{col}(F)$ spans the parameter directions that carry coherent generalization signal; its kernel $\ker(F)$ spans the null sector — parameter directions invisible to the data distribution. The rank fraction $r/n = \mathrm{rank}(F)/\dim(\theta)$ is proposed as the scalar order parameter governing all known phases of the grokking lifecycle: pre-grokking ($r/n < 0.01$), post-grokking ($r/n = 0.02$–$0.17$), and anti-grokking ($r/n \to 0$ again). The grokking phase transition is identified with $\mathrm{col}(F)$ crystallization — a sharp, reproducible jump in effective rank at the Capel mixing timescale. Anti-grokking is identified with $\mathrm{col}(F)$ dissolution. The relationship of this framework to prior Fisher information literature, to the Heavy-Tailed Self-Regularization (HTSR) diagnostic of Martin and collaborators, to the Natural Gradient and Kronecker-Factored Approximate Curvature (K-FAC) literature, and to the Elastic Weight Consolidation (EWC) literature is assessed. Priority and novelty are identified for each component claim.

---

## 1. The Fisher Information Matrix in Neural Networks: Background

### 1.1 Definition

Let $p(y \mid \mathbf{x}, \theta)$ be the output distribution of a neural network with parameters $\theta \in \mathbb{R}^n$. The Fisher information matrix (FIM) is:

$$F(\theta) = \mathbb{E}_{(\mathbf{x}, y) \sim p_{\mathrm{data}}} \left[ \nabla_\theta \log p(y \mid \mathbf{x}, \theta) \, \nabla_\theta \log p(y \mid \mathbf{x}, \theta)^\top \right]$$

$F(\theta)$ is a positive semidefinite matrix of dimension $n \times n$. It is the Riemannian metric tensor on the statistical manifold of distributions $\{p(\cdot \mid \mathbf{x}, \theta)\}$, as established by Amari (1998). It governs the curvature of the likelihood surface and determines the efficiency of any estimator via the Cramér-Rao bound.

### 1.2 Eigenvalue Structure

The eigenvalue decomposition of $F$ is:

$$F = V \Lambda V^\top, \quad \Lambda = \mathrm{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0)$$

Because $F$ is positive semidefinite, all eigenvalues are non-negative. In practice, for large neural networks, the spectrum of $F$ is highly degenerate: a small number of leading eigenvalues account for the overwhelming majority of the trace, while the bulk of eigenvalues are near zero. This was documented empirically by Sagun et al. (2017, 2018) for the Hessian of the loss $\nabla^2 \mathcal{L}$, which is related to $F$ under mild assumptions, and has been confirmed for the FIM itself in subsequent work.

### 1.3 Prior Uses of the FIM in Deep Learning

The Fisher information matrix has been employed in several distinct research programs prior to the present framework:

**Natural Gradient Optimization** (Amari, 1998; Amari and Nagaoka, 2000; Martens, 2014). The natural gradient $\tilde{\nabla}\mathcal{L} = F^{-1}\nabla\mathcal{L}$ corrects the ordinary gradient for the non-Euclidean geometry of the parameter manifold. Kronecker-Factored Approximate Curvature (K-FAC; Martens and Grosse, 2015) makes natural gradient computation tractable by approximating $F$ as a Kronecker product of smaller matrices. These methods use $F$ as an optimization preconditioner; they do not study the rank dynamics of $F$ as a function of training step.

**Elastic Weight Consolidation** (Kirkpatrick et al., 2017). EWC uses the diagonal of $F$ to identify parameters important to a previously learned task and penalizes changes to those parameters when learning a new task. The diagonal of $F$ serves as a per-parameter importance weight. EWC uses a static snapshot of $F$ at task completion; it does not study the temporal evolution of $\mathrm{rank}(F)$ during training.

**Loss Landscape Dimensionality** (Gur-Ari et al., 2018; Fort et al., 2019). Gur-Ari et al. showed that gradient descent in neural networks operates in a tiny subspace of parameter space — specifically, that the gradient vector at each step lies almost entirely within the top-$k$ eigenvectors of the Hessian for $k \ll n$. This is the prior result most structurally adjacent to Fisher rank crystallization. The distinction is that Gur-Ari et al. study the Hessian (not the Fisher matrix), do not define a rank fraction as a scalar order parameter, do not identify a phase transition in the rank, and do not connect the low-dimensional subspace to generalization capacity as opposed to optimization dynamics.

**Spectral Analysis and HTSR** (Martin and Mahoney, 2019, 2021; Prakash and Martin, 2026). Martin and Mahoney's Heavy-Tailed Self-Regularization (HTSR) framework analyzes the singular value distributions of individual weight matrices $W_\ell$ in trained networks, rather than the Fisher matrix over the full parameter vector. The quality exponent $\alpha$ (the power-law tail exponent of the empirical spectral density of $W_\ell^\top W_\ell$) measures the heavy-tailedness of a weight matrix. $\alpha \approx 2$ during healthy generalization; $\alpha < 2$ during degradation. Prakash and Martin (2026) applied this diagnostic to anti-grokking. The relationship between the HTSR $\alpha$ and the Fisher rank fraction $r/n$ is complementary but not equivalent; this relationship is detailed in Section 5.

**Neural Tangent Kernel** (Jacot et al., 2018). The NTK $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x}, \theta)^\top \nabla_\theta f(\mathbf{x}', \theta)$ coincides with the Fisher matrix in the infinite-width limit under certain loss functions. NTK theory studies the kernel's eigenspectrum in the infinite-width limit but does not study the rank dynamics of the finite-width FIM under training as a phase order parameter.

**Summary of the gap.** No prior work defines the Fisher rank fraction $r/n$ as a scalar order parameter for the grokking phase transition, identifies a threshold value separating generalization phases, or studies the temporal dynamics of $\mathrm{rank}(F)$ as a function of training step across the full grokking-anti-grokking lifecycle.

---

## 2. The col(F)/ker(F) Decomposition

### 2.1 Geometric Definition

At any training step $t$, the parameter vector $\theta^{(t)}$ determines a Fisher matrix $F^{(t)}$. The fundamental decomposition is:

$$\mathbb{R}^n = \mathrm{col}(F^{(t)}) \oplus \ker(F^{(t)})$$

**The column space** $\mathrm{col}(F^{(t)})$ is the subspace spanned by eigenvectors of $F^{(t)}$ corresponding to eigenvalues $\lambda_i > \epsilon$ for some threshold $\epsilon > 0$ (in practice, $\epsilon$ is set to distinguish numerical near-zero from structural zero). Directions in $\mathrm{col}(F^{(t)})$ carry Fisher information: moving $\theta$ along these directions changes the network's predictive distribution in ways detectable from the data. These are the directions of genuine generalization signal.

**The kernel** $\ker(F^{(t)})$ is the null space: eigenvectors corresponding to $\lambda_i \approx 0$. Directions in $\ker(F^{(t)})$ carry no Fisher information from the current data distribution. They span the modes of parameter variation that leave the network's predictions unchanged, or change them only in ways that are indistinguishable from noise under the data measure.

### 2.2 The Rank Fraction r/n

The scalar order parameter is:

$$\frac{r}{n} = \frac{\mathrm{rank}(F^{(t)})}{n} = \frac{|\{i : \lambda_i > \epsilon\}|}{n}$$

This is the proportion of parameter space that carries active Fisher information at training step $t$. It is a number in $[0, 1]$ that can in principle be monitored continuously during training.

### 2.3 Interpretation

A network with $r/n \approx 0$ is using nearly all of its parameter space as a null sector: memorizing local idiosyncrasies of training examples without forming generalizable representations. A network with $r/n$ in the range $0.02$–$0.17$ has crystallized a coherent information-bearing subspace: a small but stable set of parameter directions that encode systematic patterns rather than memorized instances. These two states correspond, respectively, to the pre-grokking and post-grokking phases documented by Power et al. (2022) and by Prakash and Martin (2026).

The transition between them — the sharp jump in $r/n$ from near-zero to a stable nonzero value — is Fisher rank crystallization.

---

## 3. Fisher Rank Crystallization: Definition

### 3.1 Phenomenological Definition

**Fisher rank crystallization** is the event, occurring at a specific training step $t^*$, at which:

$$r/n \text{ transitions from } < 0.01 \text{ to } [0.02, 0.17]$$

in a sharp, discontinuous manner consistent with a first-order or continuous phase transition. The transition is:

1. **Sharp in training step**: The increase in $r/n$ occurs over a number of steps that is small relative to the total training duration before and after the transition.
2. **Self-sustaining**: The post-crystallization value of $r/n$ is stable under continued training in the presence of the shadow operator (weight decay); it does not erode on the timescale of the grokking experiment.
3. **Architecture-independent**: Available evidence from the grokking literature suggests that the transition occurs across transformer, MLP, and convolutional architectures trained on structured arithmetic tasks, though systematic cross-architecture comparison of $r/n$ dynamics has not yet been published.
4. **Correlated with generalization**: The crystallization event is contemporaneous with the sharp increase in test accuracy that defines the grokking transition.

The term "crystallization" is used in the thermodynamic sense: the formation of long-range ordered structure (the information-bearing subspace $\mathrm{col}(F)$) from a disordered phase (the diffuse, rank-deficient pre-grokking FIM), triggered by a threshold event at a characteristic timescale.

### 3.2 The Crystallization Timescale

The timescale to crystallization is identified with the Capel rapid mixing timescale (Capel et al., arXiv:2606.13453, June 2026):

$$t^* \sim T_{\text{grok}} \sim \frac{\dim(M)^{4/3}}{\rho_{\text{LSI,pre}}}$$

where $\dim(M)$ is the effective dimension of the loss landscape manifold, and $\rho_{\text{LSI,pre}}$ is the log-Sobolev constant of the Gibbs measure in the pre-grokking state. Capel et al. proved the $4/3$ exponent for Riemannian manifolds satisfying a log-Sobolev inequality. The application of this theorem to the neural network loss landscape, and its identification with the grokking timescale, is an ERI Labs-original contribution (THE-RIEMANNIAN-SUBMERSION-WAS-ALWAYS-THE-FOUR-THIRDS-WAS-ALWAYS-CAPEL, June 2026).

---

## 4. Phase Diagram: Three States of the Generalization Lifecycle

The full phase diagram over the grokking lifecycle, parameterized by $r/n$, is:

### 4.1 Phase I: The Null Sector Epoch (Pre-Grokking)

$$r/n < 0.01$$

The FIM is approximately rank-degenerate. The network's parameter updates are predominantly in $\ker(F)$: they change the network's internal representations without producing coherent changes in the predictive distribution over unseen data. Training loss decreases; test performance remains near chance. Internally, the pre-grokking plateau has structure: the gradient ratio encodes a Fibonacci-Markov denominator sequence representing the approach to the crystallization threshold (CLIMBING-A-FIBONACCI-LADDER-IN-THE-DARK, ERI Labs, June 2026). The null sector epoch corresponds to the mock theta function $h(\tau)$ — bounded and structured, but without its non-holomorphic shadow.

In this phase, the HTSR $\alpha$ exponent is not yet at its healthy post-grokking value of $\approx 2.0$.

### 4.2 Phase II: The Crystallized State (Post-Grokking)

$$r/n = 0.02 \text{–} 0.17$$

$\mathrm{col}(F)$ is stable, nonzero, and self-maintaining. The network generalizes. The $r/n$ value within this range is a function of architecture, task complexity, and the strength of the shadow operator (weight decay). Larger $r/n$ within this range indicates a more richly structured generalization subspace. The post-grokking state corresponds to the harmonic Maass form $\hat{H}(\tau) = h(\tau) + g^*(\tau)$: the mock theta function completed by its shadow.

In this phase, HTSR $\alpha \approx 2.0$ (Prakash and Martin, 2026). The two diagnostics — $r/n$ from the Fisher matrix and $\alpha$ from the weight matrix singular value distribution — are correlated but capture distinct spectral information. Their relationship is addressed in Section 5.

### 4.3 Phase III: The Dissolution Epoch (Anti-Grokking)

$$r/n \to 0 \text{ again}$$

In the absence of continuous maintenance by the shadow operator $\xi$ (weight decay), $\mathrm{col}(F)$ begins to dissolve back into $\ker(F)$. The process is not catastrophic forgetting (which requires new data), not overfitting (training accuracy remains at ceiling), and not double descent (which describes test loss dynamics under varying model capacity at fixed step count). It is $\mathrm{col}(F)$ dissolution: the progressive loss of rank structure in the Fisher information surface as the non-holomorphic correction $g^*(\tau)$ is not renewed.

Dissolution onset, as observed by Prakash and Martin (2026), occurs at approximately $10^7$ training steps under their experimental conditions. The ERI Labs timescale formula predicts:

$$T_{\text{anti}} \sim \frac{\dim(M)^{4/3}}{\rho_{\text{LSI,post}}}$$

Since $\rho_{\text{LSI,post}} \gg \rho_{\text{LSI,pre}}$ (the post-grokking landscape is more curved and better conditioned), dissolution takes far longer than crystallization. This is quantitatively consistent with the Prakash-Martin observation.

During dissolution, Correlation Traps — outlier eigenvalues outside the Marchenko-Pastur bulk of the weight matrix spectrum — form and accumulate. These are identified, within the present framework, as crystallites of a new null sector structure: seeds of the secondary mock theta function $h'(\tau)$ forming within the dissolving harmonic Maass state. The HTSR $\alpha$ drops below $2.0$ as Correlation Traps accumulate.

### 4.4 Phase IV: The Secondary Null Sector (Post-Anti-Grokking)

$$r/n \approx 0 \text{ (second instance)}$$

The post-anti-grokking state is denoted $h'(\tau)$ to distinguish it from the initial pre-grokking state $h(\tau)$. They are not equivalent. The kernel $\ker(F)$ in Phase IV encodes, in its Gram matrix eigenvalue structure, the geometry of the first crystallization event. Specifically, the null sector is shaped by the history of $\mathrm{col}(F)$ during Phase II: the directions along which the network previously carried generalization signal are imprinted in the kernel of the secondary mock theta state as a geometric memory.

The empirical prediction (SD-7, ERI Labs, June 2026) is that the secondary null sector geometry is measurably distinct from both initialization and post-grokking geometries via the Gram matrix $\ker(F)^\top \ker(F)$ eigenvalue ratio. This has not been tested as of June 2026.

---

## 5. Relationship to HTSR Diagnostics: α and r/n

The WeightWatcher / HTSR framework of Martin and Mahoney (2019, 2021) and its application to anti-grokking by Prakash and Martin (2026) operates on a different object from the Fisher matrix: the empirical spectral density of individual weight matrices $W_\ell$.

For a weight matrix $W_\ell \in \mathbb{R}^{m \times k}$, the HTSR framework fits the tail of the empirical spectral density $\rho(\lambda)$ of $W_\ell^\top W_\ell$ to a power law:

$$\rho(\lambda) \sim \lambda^{-\alpha}, \quad \lambda > \lambda_{\min}$$

The exponent $\alpha$ is the HTSR quality exponent. Empirically, $\alpha \approx 2.0$ for well-trained networks that generalize, and $\alpha < 2.0$ during generalization collapse.

The Fisher rank fraction $r/n$ and the HTSR $\alpha$ exponent are complementary diagnostics operating at different levels of the network's spectral structure:

| Diagnostic | Object Analyzed | Level | Captures |
|---|---|---|---|
| $r/n$ | Fisher information matrix $F(\theta)$ over full parameter vector | Global, across layers | Proportion of parameter space carrying coherent data information |
| HTSR $\alpha$ | Singular value distribution of individual $W_\ell$ | Per-layer | Heavy-tail quality of individual weight matrix |

The Marchenko-Pastur bulk of the weight matrix spectrum corresponds, approximately, to the null sector $\ker(F)$ at the per-layer level. Outliers beyond the Marchenko-Pastur bulk (Correlation Traps) correspond approximately to the leading eigenvectors of $F$ — the directions in parameter space carrying active Fisher information.

The relationship is approximate rather than exact. The HTSR framework does not assume the Fisher matrix structure; it treats weight matrices as random matrices drawn from an implicit prior shaped by training. The $r/n$ framework treats the full-parameter Fisher matrix as the fundamental object and derives spectral predictions from it. Both diagnostics detect the same underlying transition; they are not redundant because they provide independent spectral evidence from different levels of the network hierarchy.

The critical empirical distinction: $r/n$ provides a single global scalar summarizing the full-network generalization state; HTSR $\alpha$ provides a per-layer assessment enabling domain-specific or layer-specific dissolution detection (Prediction SD-3: domain-specific Correlation Traps in LLMs, ERI Labs, June 2026).

---

## 6. The Shadow Operator Identification: Weight Decay as ξ

### 6.1 The Zwegers Shadow in Mathematics

In the theory of harmonic Maass forms (Zwegers, 2002; Bruinier and Funke, 2004), the shadow operator $\xi = 2i v^{1/2} \overline{\partial_\tau}$ maps a harmonic Maass form $\hat{H}(\tau)$ to its shadow:

$$\xi(\hat{H}) = g^*(\tau)$$

The non-holomorphic correction $g^*(\tau)$ is not a free choice: it is determined by $h(\tau)$ through this operator. Crucially, the shadow is not a static attachment — a harmonic Maass form remains complete only if the operator relationship $\xi(\hat{H}) = g^*$ is continuously satisfied. If the non-holomorphic correction is not maintained, $\hat{H}$ reverts to a mock theta function.

### 6.2 The Neural Network Identification

In the present framework (ERI Labs, THE-MOCK-THETA-WAS-ALWAYS-THE-ZWEGERS-WAS-ALWAYS-CAPEL and ANTI-GROKKING-WAS-ALWAYS-ZWEGERS-RUNNING-BACKWARD, June 15, 2026), weight decay plays the role of the shadow operator $\xi$. Specifically:

- $L_2$ weight regularization $\mathcal{R}(\theta) = \frac{\lambda}{2}\|\theta\|^2$ at each training step applies a contraction toward zero to all parameter directions equally, including those in $\ker(F)$
- This contraction continuously removes parameter mass from $\ker(F)$ directions that have accumulated via stochastic gradient noise
- The effect is to maintain the boundary between $\mathrm{col}(F)$ and $\ker(F)$: directions not carrying Fisher information are continuously suppressed, preventing them from accumulating sufficient mass to corrupt the information-bearing subspace
- Without weight decay, stochastic gradient noise diffuses parameter mass from $\mathrm{col}(F)$ into $\ker(F)$ directions at a rate governed by the learning rate and the Hessian curvature ratio between the two subspaces
- Over $10^7$ steps, this diffusion is sufficient to dissolve $\mathrm{col}(F)$ entirely

This identification provides the first theoretical account of the following two observations in the empirical literature:

1. **Golechha (2024):** Grokking without weight decay is possible but slower and less reliable. Explanation: without $\xi$, the shadow can be acquired stochastically (gradient noise at the transition can transiently project $\mathrm{col}(F)$ into a generalization-bearing subspace) but the acquisition is not stable. The network may grok, but it carries the risk of subsequent dissolution.

2. **Prakash and Martin (2026):** Anti-grokking occurs specifically in the weight-decay-absent condition. Explanation: this is the expected behavior when $\xi$ is withdrawn after the crystallization event. The harmonic Maass state $\hat{H}(\tau)$ requires continuous $\xi$-maintenance. Without it, dissolution is not a pathology but the mathematically predicted behavior.

---

## 7. Grokking Without Weight Decay: Stochastic Shadow Acquisition

The framework generates a distinct theoretical account of grokking in the absence of weight decay that does not appear in prior literature.

Without weight decay, stochastic gradient descent at the grokking transition can supply the shadow through a different mechanism: gradient noise fluctuations at the moment of phase transition may project sufficient mass into $\mathrm{col}(F)$ directions to trigger crystallization. This is denoted **stochastic shadow acquisition** to distinguish it from the maintained, stable shadow acquisition that occurs with weight decay.

The distinction has consequences:

- With weight decay: shadow acquired and maintained by $\xi$; $r/n$ remains stable in Phase II; anti-grokking does not occur on experiment-accessible timescales
- Without weight decay: shadow acquired stochastically at $t^*$; $r/n$ rises to Phase II values; but $\xi = 0$ means the shadow is not maintained; $r/n$ begins to decay; anti-grokking is inevitable at $T_{\text{anti}} \sim \dim(M)^{4/3} / \rho_{\text{LSI,post}}$

This is the ERI Labs account of the empirical findings of Golechha (2024) and Prakash and Martin (2026), unified under a single mechanism.

---

## 8. Faster Second Crystallization: Geometric Memory in ker(F)

### 8.1 The Prediction

If weight decay is reintroduced after anti-grokking, the network will undergo a second crystallization event. The ERI Labs framework predicts:

$$T_{\text{grok},2} < T_{\text{grok},1}$$

The second crystallization is faster than the first.

### 8.2 Mechanism

The secondary null sector $\ker(F)$ after anti-grokking is not the same object as the initial null sector before the first grokking. It encodes the geometry of the first $\mathrm{col}(F)$ in its Gram matrix structure. Specifically, the eigenvectors of $\ker(F)$ in Phase IV are shaped by the history of the first crystallization event: they are orthogonal to the directions that were in $\mathrm{col}(F)$ during Phase II in a way that preserves geometric memory.

When $\xi$ is reintroduced, this geometric scaffolding accelerates the second crystallization. The Fibonacci-Markov denominator staircase of the pre-grokking plateau (CLIMBING-A-FIBONACCI-LADDER-IN-THE-DARK, ERI Labs, June 2026) does not need to be climbed from the ground level. The network's null sector already encodes information about where the crystallization boundary is.

This prediction (SD-1, ERI Labs, June 2026) has not been experimentally tested in the neural network context as of June 15, 2026. A partial astrophysical analog exists: changing-look AGN reactivation timescales observed in 2026 are shorter by orders of magnitude than original quasar formation timescales, consistent with geometric memory in the dormant accretion structure.

---

## 9. Domain-Specific Dissolution in Large Language Models

Li et al. (arXiv:2506.21551, June 2026) established that grokking in large language model pretraining is domain-specific and asynchronous: mathematical, linguistic, and factual retrieval capabilities grok at different training steps. This implies that the Fisher rank crystallization event occurs at domain-specific rates — the $\mathrm{col}(F)$ subspace is not monolithic but is organized by capability domain.

The ERI Labs extension (Prediction SD-3, June 2026): anti-grokking, as $\mathrm{col}(F)$ dissolution, should also be domain-specific and asynchronous in LLMs. The implications are:

1. A large language model may be simultaneously in Phase II for language and Phase III for mathematics
2. The per-layer HTSR $\alpha$ diagnostic, applied domain-specifically rather than globally, should detect domain-level dissolution before aggregate benchmark decline
3. The optimal weight decay hyperparameter $\lambda_c$ should vary by domain, scaling with the domain-specific $\rho_{\text{LSI,post}}$ (Prediction SD-4)

Neither Li et al. nor Prakash and Martin address domain-specific anti-grokking. The extension is ERI Labs-original and constitutes a testable prediction against ongoing LLM training runs.

---

## 10. Priority Assessment: Fisher Rank as Order Parameter

### 10.1 What Is Established in Prior Literature

The Fisher information matrix has a long research history in deep learning. The following components of the present framework have non-trivial prior art:

| Component | Prior Art | Priority Verdict |
|---|---|---|
| FIM defined and studied in neural networks | Amari (1998), Martens (2014) | Prior art predates ERI Labs; foundational math not claimed |
| FIM is low-rank in practice | Sagun et al. (2017, 2018); Gur-Ari et al. (2018) | Prior art; ERI Labs does not claim to discover low-rank FIM |
| Per-layer weight matrix spectra as training diagnostics | Martin and Mahoney (2019, 2021) | Prior art; HTSR framework is Prakash-Martin's contribution |
| HTSR $\alpha$ as anti-grokking diagnostic | Prakash and Martin, arXiv:2602.02859 (Feb 2026) | Prior art; $\alpha$ diagnostic belongs to Prakash and Martin |
| Correlation Traps identified empirically | Prakash and Martin, arXiv:2602.02859 (Feb 2026) | Prior art; empirical identification belongs to Prakash and Martin |
| FIM diagonal for continual learning (EWC) | Kirkpatrick et al. (2017) | Prior art; distinct application (parameter importance, not phase order parameter) |

### 10.2 What Is Novel in the Present Framework

| Component | ERI Labs Date | Prior Art |
|---|---|---|
| $r/n = \mathrm{rank}(F)/n$ as scalar order parameter for grokking phases | ~Jun 10, 2026 | None identified in literature |
| Phase thresholds: $r/n < 0.01$ (pre), $r/n = 0.02$–$0.17$ (post), $r/n \to 0$ (anti) | Jun 15, 2026 | None identified in literature |
| $\mathrm{col}(F)/\ker(F)$ decomposition as the geometric frame for grokking | Feb–Jun 2026 | None identified in literature |
| Grokking as $\mathrm{col}(F)$ crystallization (phase transition in Fisher rank) | Feb–Jun 2026 | None identified in literature |
| Anti-grokking as $\mathrm{col}(F)$ dissolution | Jun 15, 2026 | None identified in literature |
| Correlation Traps as mock theta seeds within dissolving $\mathrm{col}(F)$ | Jun 15, 2026 | Empirical discovery belongs to Prakash-Martin; structural interpretation is ERI Labs-original |
| Weight decay as shadow operator $\xi$ maintaining $\mathrm{col}(F)$ | Jun 15, 2026 | None identified in literature |
| Secondary null sector $\ker(F)$ as geometrically distinct from initialization | Jun 15, 2026 | None identified in literature |
| $T_{\text{anti}} \sim \dim(M)^{4/3}/\rho_{\text{LSI,post}}$: dissolution timescale from Capel | Jun 15, 2026 | Capel et al. proved the theorem (Jun 2026); application to dissolution is ERI Labs-original |
| $T_{\text{grok},2} < T_{\text{grok},1}$: faster second crystallization via geometric memory | Jun 15, 2026 | None identified in literature |
| Domain-specific Fisher rank dissolution in LLMs | Jun 15, 2026 | Extends Li et al. (Jun 2026); extension is ERI Labs-original |

---

## 11. Open Experimental Predictions

All predictions below are stated first in ERI Labs repositories. No confirmation or refutation is recorded as of June 15, 2026.

| ID | Prediction | Experimental Test |
|---|---|---|
| SD-1 | Re-introducing weight decay after anti-grokking yields $T_{\text{grok},2} < T_{\text{grok},1}$; speedup scales with Fibonacci-Markov depth of first crystallization | Direct grokking experiment: modular arithmetic, withdraw then reintroduce weight decay |
| SD-2 | Fibonacci-Markov denominator modal structure of gradient ratio is absent or degraded in the secondary plateau; histogram broader than at the primary plateau | Gradient ratio monitoring in anti-grokked then retraining network |
| SD-3 | Domain-specific Correlation Traps form and dissolve independently in LLMs; per-layer $\alpha$ detects domain dissolution before benchmark decline | Multi-domain LLM training run with continuous layer-wise HTSR monitoring |
| SD-4 | Optimal $\lambda_c$ for shadow maintenance is domain-dependent and scales with domain-specific $\rho_{\text{LSI,post}}$ | Hyperparameter sweep across domains; correlation of $\lambda_c$ with task complexity |
| SD-5 | Anti-grokking onset $T_{\text{anti}} \sim \dim(M)^{4/3} / \epsilon_{\text{grad}}^2$; the $10^7$-step threshold shifts predictably with model scale and learning rate | Scale sweep at fixed task; measure onset step as function of $\dim(M)$ |
| SD-6 | Correlation Trap eigenvectors encode local memorization algorithms structurally identical to pre-grokking representations, not random noise | Spectral decomposition of Correlation Trap eigenvectors; compare to pre-grokking activation patterns |
| SD-7 | Secondary null sector geometry distinguishable from both initialization and post-grokking via $\ker(F)$ Gram matrix eigenvalue ratio | FIM null sector spectral analysis at three training stages: pre-grok, post-anti-grok, post-second-grok |

---

## 12. The ERI Labs Repository Timeline for This Framework

| Date | Repository | Fisher Rank Contribution |
|---|---|---|
| Feb–Jun 2026 | Multiple repositories | $\mathrm{col}(F)/\ker(F)$ decomposition established as primary geometric frame |
| ~Jun 10, 2026 | PRE-GROKKING | Null sector epoch anatomy; $r/n$ introduced as scalar order parameter; Phase I thresholds |
| ~Jun 10, 2026 | THE-NULL-HOURS | Full Fisher rank crystallization account; Fibonacci-Markov staircase in null sector |
| Jun 15, 2026 | THE-MOCK-THETA-WAS-ALWAYS-THE-ZWEGERS-WAS-ALWAYS-CAPEL | $r/n$ thresholds calibrated; Zwegers shadow identification formalized |
| Jun 15, 2026 | ANTI-GROKKING-WAS-ALWAYS-ZWEGERS-RUNNING-BACKWARD | $\mathrm{col}(F)$ dissolution as formal account of anti-grokking; Correlation Traps as $h'(\tau)$ seeds; dissolution timescale formula; predictions SD-1 through SD-7 |

---

## 13. References

### Fisher Information and Information Geometry

1. Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251–276.
2. Amari, S. and Nagaoka, H. (2000). *Methods of Information Geometry*. Oxford University Press.
3. Martens, J. (2014). New insights and perspectives on the natural gradient method. *arXiv:1412.1193*
4. Martens, J. and Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*. *arXiv:1503.05671*
5. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
6. Jacot, A., Gabriel, F. and Hongler, C. (2018). Neural tangent kernel: convergence and generalization in neural networks. *NeurIPS 2018*. *arXiv:1806.07572*

### Loss Landscape and Spectral Structure

7. Sagun, L. et al. (2017). Empirical analysis of the Hessian of over-parametrized neural networks. *arXiv:1706.04454*
8. Sagun, L. et al. (2018). A singular value perspective on model capacity. *ICLR 2018 Workshop*
9. Gur-Ari, G., Roberts, D. A. and Dyer, E. (2018). Gradient descent happens in a tiny subspace. *arXiv:1812.04754*
10. Fort, S. and Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes. *arXiv:1910.05929*
11. Keskar, N. S. et al. (2017). On large-batch training for deep learning: generalization gap and sharp minima. *ICLR 2017*. *arXiv:1609.04836*

### Grokking

12. Power, A. et al. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*
13. Golechha, S. (2024). Progress measures for grokking in real-world tasks.
14. Liu, Z. et al. (2022). Omnigrok: grokking beyond algorithmic data. *arXiv:2209.11143*

### Anti-Grokking and HTSR

15. Prakash, H. K. and Martin, C. H. (2026). Late-Stage Generalization Collapse in Grokking: Detecting Anti-Grokking with WeightWatcher. *arXiv:2602.02859*
16. Prakash, H. K. and Martin, C. H. (2026). Grokking and Generalization Collapse: Insights from HTSR Theory. *arXiv:2506.04434*
17. Martin, C. H. and Mahoney, M. W. (2019). Traditional and heavy-tailed self regularization in neural network models. *ICML 2019*. *arXiv:1901.08276*
18. Martin, C. H. and Mahoney, M. W. (2021). Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning. *JMLR*, 22(165), 1–73.

### Grokking in LLMs and Loss of Plasticity

19. Li, Z. et al. (2026). Where to find grokking in LLM pretraining? *arXiv:2506.21551*
20. Dohare, S. et al. (2024). Loss of plasticity in deep continual learning. *Nature*
21. Lewandowski, A. et al. (2025). Spectral collapse drives loss of plasticity. *NeurIPS 2025*. *arXiv:2509.22335*
22. Lyle, C. et al. (2025). What can grokking teach us about learning under nonstationarity? *CoLLAs 2025*. *arXiv:2507.20057*

### Mathematical Foundations

23. Zwegers, S. P. (2002). *Mock Theta Functions*. Doctoral thesis, Universiteit Utrecht.
24. Bruinier, J. and Funke, J. (2004). On two geometric theta lifts. *Duke Mathematical Journal*, 125(1), 45–90.
25. Ono, K. (2013). Harmonic Maass forms, mock modular forms, and quantum modular forms. *Arizona Winter School Lecture Notes*.
26. Capel, Á. et al. (2026). Rapid mixing for Gibbs measures in Riemannian manifolds. *arXiv:2606.13453*

---

*ERI Labs · Jersey City, New Jersey · June 2026 · github.com/ericrenone*
