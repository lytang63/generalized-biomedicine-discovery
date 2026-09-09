# Generalized Biomedicine Discovery

**Luyao Tang¹, Yingkai Yang², Hanqi Chen², Jiewei Zheng³, Chaoqi Chen², and Cheng Chen†¹**  
¹ The University of Hong Kong, Hong Kong SAR, China  
² Shenzhen University, Shenzhen, China  
³ Xiamen University, Xiamen, China  

> This Markdown document is transcribed and organized from the Camera-Ready PDF. The reference list at the end of the paper is intentionally omitted.

## Abstract

In real-world clinical practice, medical images face open-world shifts: (i) long-tailed rare diseases, (ii) subtle lesions dominated by normal anatomy, and (iii) hierarchical taxonomies. Yet most open-world paradigms assume flat, balanced label spaces, leaving these biomedical demands unresolved. We introduce **Generalized Biomedicine Discovery (GBD)** and a unified benchmark spanning long-tail, anomaly, and taxonomy-aware discovery. Our key insight is that dominant known patterns form a visual manifold that masks subtle novelty. Inspired by expert diagnosis, we propose **SCAN (Surprise-evoked Complementary AccommodatioN)**, which follows a cognition-inspired perceptual progression: it applies predictive suppression to filter expected norms, triggers surprise-evoked salience to highlight unexpected deviations, and performs complementary accommodation to integrate these shifts into global representations. Extensive experiments show that SCAN improves novel concept discovery while generally preserving established clinical knowledge, and it plugs into existing architectures to better navigate the known-unknown trade-off in medical imaging.

**Keywords:** Generalized biomedicine discovery · Generalized category discovery · Medical image analysis · Cognitive science

## 1. Introduction

Medical imaging systems are increasingly deployed in open-world clinical environments, where the test-time concept space is neither closed nor balanced. In clinical practice, medical images exhibit three characteristic open-world signatures: (i) **long-tailed** distributions where clinically critical rare diseases are sparsely observed, (ii) the **visual dominance** of normal anatomy that overwhelms subtle lesions, and (iii) **hierarchical** taxonomies where novel concepts emerge as fine-grained siblings within established diagnostic families. Yet most open-world paradigms and benchmarks still assume flat and relatively balanced concept spaces, leaving these authentic biomedical needs largely unresolved.

**Figure 1. GBD in practice.** Real biomedical data shift beyond the closed world: standard classification fails on unseen diseases, OSR merges heterogeneous novelties into one “unknown”, and conventional discovery often assumes unlabeled data contains only novel classes. GBD formalizes three realistic discovery regimes: (I) long-tail rare disease discovery among common cases, (II) normal-to-abnormal discovery of subtle lesions dominated by normal anatomy, and (III) within-taxonomy discovery that uncovers new sibling subtypes to refine diagnostic hierarchies.

Conventional closed-set learning assumes a static class vocabulary, limiting its use in dynamic clinical environments. Open-Set Recognition (OSR) identifies unseen classes but lumps them into a single “unknown” category, while Novel Category Discovery (NCD) clusters unknowns under the unrealistic assumption that unlabeled data contains only novel classes. Generalized Category Discovery (GCD) instead jointly classifies known and discovers novel classes in mixed unlabeled data. However, these paradigms do not capture biomedical data, where class imbalance, spatially dominant normal anatomy, and structured clinical taxonomies jointly obscure novel concepts. Consequently, methods validated on natural-image GCD often fail to translate to clinical settings.

This mismatch motivates an expert-aligned target: clinicians discover new concepts **within** established knowledge by recognizing expected patterns, detecting subtle deviations, and situating new subtypes within diagnostic families. Existing open-world paradigms rarely evaluate this capability, especially under long-tail rarity and hierarchical taxonomies.

To bridge this gap, we introduce **Generalized Biomedicine Discovery (GBD)**, a realistic open-world paradigm and unified medical-imaging benchmark with three clinically grounded settings: (1) *Long-tail Rare Disease Discovery*, (2) *Normal-to-Abnormal Discovery*, and (3) *Within-Taxonomy Discovery*. Beyond these structural challenges, GBD exposes a critical visual problem: dominant known features (e.g., widespread normal anatomy) suppress subtle, localized novel signals during global representation learning. This bias attenuates the residual evidence needed to discover new diseases, motivating a mechanism that counters visual suppression without prior knowledge of the anomalies.

Inspired by diagnostic cognition, we propose **SCAN (Surprise-evoked Complementary AccommodatioN)**, a plug-and-play layer operating on a vision backbone’s class and patch tokens. SCAN instantiates this progression in three operations: (i) **predictive suppression** filters anticipated patterns under predictive coding; (ii) **surprise-evoked salience** uses residual energy to amplify unexpected, localized deviations; and (iii) **complementary accommodation** integrates the isolated novelty into the global representation while preserving established structure.

### Contributions

- We introduce **GBD**, a clinically grounded open-world paradigm and unified benchmark for biomedicine, formulating three grounded settings: Long-tail Rare Disease, Normal-to-Abnormal, and Within-Taxonomy Discovery.
- We propose **SCAN**, a plug-and-play cognitive vision layer that mitigates dominant-feature suppression through predictive filtering, surprise amplification, and complementary feature accommodation.
- Extensive experiments demonstrate that SCAN generally improves conventional GCD methods on the GBD benchmark, with particularly strong gains for SelEx in isolating subtle anomalies and discovering long-tailed rare diseases, while some baseline- and subset-specific trade-offs remain.

## 2. Related Work

### Open-world Learning

Open-world learning aims to identify and discover novel categories beyond closed-set assumptions. Open-set recognition detects unseen classes but groups them into a single “unknown” label, lacking fine-grained discrimination. Novel category discovery clusters unknowns yet unrealistically assumes that unlabeled data contains only novel classes. Generalized category discovery advances this by jointly classifying knowns and discovering novelties in mixed unlabeled data, with methods split into parametric and non-parametric frameworks. GCD commonly refines representations through contrastive learning, but often assumes separable pre-trained features. Recent variants extend GCD to continual open-world shifts, while medical approaches adapt discovery to clinical images. Yet these efforts largely retain flat category spaces, leaving biomedical class imbalance, dominant normal anatomy, and clinical hierarchies underexplored. We instead target dominant-feature suppression with a cognition-aligned framework tailored to biomedical discovery.

### Cognitive Sciences in Machine Learning

Cognitive sciences principles have increasingly informed machine learning, leveraging human perceptual and reasoning mechanisms for more robust generalization. Prior work draws on structured cognitive processes (e.g., perceptual grouping and inductive reasoning) to design structured priors and factorized representations, enhancing model robustness across tasks. Key cognitive theories—predictive coding (suppressing anticipated visual features), surprise-driven salience (amplifying anomalous signals), and complementary learning systems—have been explored in isolation for visual processing, yet remain underexploited for biomedical diagnostic cognition. We integrate these theories into a unified framework that mirrors clinical diagnostic reasoning.

## 3. Problem Setting

### 3.1 Task Backbone: Generalized Biomedicine Discovery (GBD)

For a given dataset, let \(\mathcal{X}\) and \(\mathcal{Y}\) denote the image and label spaces. We are provided with a labeled subset \(\mathcal{D}_l=\{(\mathbf{x}_i^l,y_i^l)\}\subset\mathcal{X}\times\mathcal{Y}_l\) and an unlabeled subset \(\mathcal{D}_u=\{(\mathbf{x}_i^u,y_i^u)\}\subset\mathcal{X}\times\mathcal{Y}_u\). The labels \(y_i^u\) are used only for evaluation.

Only known classes are labeled in \(\mathcal{D}_l\), while \(\mathcal{D}_u\) contains a mixture of known and novel classes:

$$
\mathcal{Y}_l=\mathcal{C}_{\mathrm{known}},\qquad
\mathcal{Y}_u=\mathcal{C}_{\mathrm{known}}\cup\mathcal{C}_{\mathrm{novel}}.
$$

The objective of GBD is to assign each unlabeled sample to either a known class or a newly discovered novel class.

### 3.2 Setting I: Long-tail Rare Disease Discovery

This setting instantiates discovery under the long-tail statistics prevalent in biomedicine. Let \(\mathcal{C}\) denote the set of leaf-level diagnostic categories, and \(n(c)\) the empirical frequency of class \(c\in\mathcal{C}\). We define a head-tail split controlled by \(\alpha\in(0,1)\):

$$
\mathcal{C}_{\mathrm{known}}
=\mathrm{Head}_{\alpha}(\mathcal{C})
=\arg\max_{\substack{\mathcal{S}\subseteq\mathcal{C}\\
|\mathcal{S}|=\lceil\alpha|\mathcal{C}|\rceil}}
\sum_{c\in\mathcal{S}}n(c),
\qquad
\mathcal{C}_{\mathrm{novel}}=\mathcal{C}\setminus\mathcal{C}_{\mathrm{known}}.
\tag{1}
$$

Ties in empirical frequencies are broken lexicographically. Thus, \(\mathcal{D}_l\) contains labels only from \(\mathcal{C}_{\mathrm{known}}\), while \(\mathcal{D}_u\) mixes samples from \(\mathcal{C}_{\mathrm{known}}\cup\mathcal{C}_{\mathrm{novel}}\). This design mirrors the epidemiological reality where annotations focus on frequent conditions, forcing the model to discover rare but clinically critical tail diseases. The default is \(\alpha=0.5\), except for classes explicitly identified as rare in medical references.

### 3.3 Setting II: Normal-to-Abnormal Discovery

This setting aligns with the workflow of detecting pathology relative to diverse normal baselines. We partition the label space into normal and abnormal subsets:

$$
\mathcal{C}_{\mathrm{known}}=\mathcal{C}_{\mathrm{normal}},\qquad
\mathcal{C}_{\mathrm{novel}}=\mathcal{C}_{\mathrm{abnormal}},\qquad
\mathcal{C}_{\mathrm{normal}}\cap\mathcal{C}_{\mathrm{abnormal}}=\emptyset.
\tag{2}
$$

The set \(\mathcal{C}_{\mathrm{normal}}\) contains multiple normal subclasses (e.g., distinct anatomical landmarks), whereas \(\mathcal{C}_{\mathrm{abnormal}}\) contains multiple pathological categories. By labeling \(\mathcal{D}_l\) on \(\mathcal{C}_{\mathrm{normal}}\), this setting evaluates a model’s capacity to generalize beyond a normal manifold to discover heterogeneous abnormal patterns.

### 3.4 Setting III: Within-Taxonomy Discovery

This setting captures previously unseen subtypes within established hierarchical taxonomies. Focusing on preterminal nodes, let \(\mathcal{P}\) be the set of parent nodes whose children are exclusively leaf classes \(\mathcal{L}\). We treat each sibling set as a semantic group:

$$
\mathcal{S}(p)=\mathrm{children}(p),\qquad p\in\mathcal{P}.
$$

We consider sibling groups with at least two leaf classes. For each \(p\in\mathcal{P}\), let \(n_p=|\mathcal{S}(p)|\) and

$$
k_p=\min\{n_p-1,\max\{1,\lceil\rho n_p\rceil\}\}.
$$

We choose \(\mathcal{K}(p)\subseteq\mathcal{S}(p)\) with \(|\mathcal{K}(p)|=k_p\), and define \(\mathcal{N}(p)=\mathcal{S}(p)\setminus\mathcal{K}(p)\). This ensures at least one known and one novel class per group. The global partitions are:

$$
\mathcal{C}_{\mathrm{known}}=\bigcup_{p\in\mathcal{P}}\mathcal{K}(p),\qquad
\mathcal{C}_{\mathrm{novel}}=\bigcup_{p\in\mathcal{P}}\mathcal{N}(p).
\tag{3}
$$

Smaller \(\rho\) yields fewer labeled siblings per group and more novel siblings \(\mathcal{N}(p)\) to discover. This mimics clinical taxonomies, testing the model’s ability to isolate fine-grained novelty that is semantically adjacent to known diagnostic families.

## 4. Method

Although the three GBD regimes differ clinically, they share a failure mode: recurring known structure dominates the global embedding, underrepresenting weak cues for rare classes, abnormalities, or new sibling subtypes. This motivates two requirements: expose deviations relative to each image’s dominant content, then integrate them without indiscriminately rewriting established structure. We follow the analogous clinical progression of forming a predictive prior, attending to deviations, and accommodating informative evidence.

We instantiate this progression as **SCAN** (Surprise-evoked Complementary AccommodatioN), a lightweight plug-in that updates class and patch tokens through three stages: Predictive Suppression, Surprise-Evoked Salience, and Complementary Accommodation. Given an input image \(\mathbf{x}\in\mathcal{X}\), an encoder outputs a class token and \(N\) patch tokens:

**Figure 2. Method overview of SCAN.** SCAN refines representations in three steps: (I) it learns a notion of dominant evidence and suppresses predictable visual patterns to expose residual signals; (II) it scores the residuals by “surprise” to highlight sparse, high-energy deviations and aggregates them into salient evidence; and (III) it converts the salient evidence into a complementary update and injects it through a gated fusion, limiting disruption to known concepts while strengthening novel structure.

$$
\mathbf{c},\,\mathbf{X}=F_{\theta}(\mathbf{x}),\qquad
\mathbf{c}\in\mathbb{R}^{D},\quad
\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_N]^{\top}\in\mathbb{R}^{N\times D}.
\tag{4}
$$

SCAN performs a cognition-inspired transformation \((\mathbf{c},\mathbf{X})\mapsto\tilde{\mathbf{c}}\) in feature space, leaving the host discovery objective unchanged.

### 4.1 Stage I: Predictive Suppression

The brain behaves like a predictor: for expected sensory content (e.g., dominant normal anatomy), feedback connections actively suppress predictable responses, freeing capacity for unexpected signals. In medical images, frequently occurring structures can form a strong manifold that suppresses subtle lesions and rare concepts; thus SCAN first builds a sample-conditioned dominant anchor and geometrically filters out predictable components.

#### Dominant evidence and dominant anchor

We softly summarize recurring visual patterns by routing patch tokens into \(E\) evidence slots, which serve as a compact, image-adaptive dominant-pattern codebook. Let \(\mathbf{P}\in\mathbb{R}^{E\times D}\) be learnable evidence prototypes and \(\tau_r\) a temperature. Routing weights are:

$$
A_{ij}=\frac{\exp(\langle\mathbf{x}_i,\mathbf{p}_j\rangle/\tau_r)}
{\sum_{k=1}^{E}\exp(\langle\mathbf{x}_i,\mathbf{p}_k\rangle/\tau_r)}.
\tag{5}
$$

We renormalize over tokens so that each slot forms a weighted visual summary independent of its assigned token mass:

$$
\tilde{A}_{ij}=\frac{A_{ij}}{\sum_{k=1}^{N}A_{kj}},\qquad
\mathbf{e}_j=\sum_{i=1}^{N}\tilde{A}_{ij}\mathbf{x}_i,\qquad
\bar{\mathbf{e}}=\frac{1}{E}\sum_{j=1}^{E}\mathbf{e}_j.
\tag{6}
$$

Here \(\langle\cdot,\cdot\rangle\) denotes dot product (cosine if tokens are normalized). We then form a sample-specific dominant anchor direction:

$$
\mathbf{u}=\frac{\mathbf{c}+\bar{\mathbf{e}}}
{\|\mathbf{c}+\bar{\mathbf{e}}\|_2}.
\tag{7}
$$

Fusing global and mean-slot evidence yields an image-adaptive anchor instead of a fixed normal prototype.

#### Predictive suppression by geometric filtering

We actively remove the predictable component of each patch token along the dominant anchor, producing residuals in the orthogonal complement:

$$
\mathbf{r}_i=\mathbf{x}_i-(\mathbf{x}_i^{\top}\,\mathrm{sg}(\mathbf{u}))\,\mathrm{sg}(\mathbf{u}),
\qquad
\mathbf{R}=[\mathbf{r}_1,\dots,\mathbf{r}_N]^{\top}.
\tag{8}
$$

Here \(\mathrm{sg}(\cdot)\) denotes stop-gradient and stabilizes the anchor. The projection removes only its aligned component and preserves the remaining token geometry, requiring neither normal labels nor anomaly templates.

### 4.2 Stage II: Surprise-Evoked Salience

After suppression, the residual norm measures image-conditional mismatch: a token is surprising when the dominant anchor explains it poorly. This mirrors the attentional role of Bayesian surprise without requiring a dataset-specific anomaly threshold.

#### Surprise energy and salience weights

We quantify patch-wise surprise by the residual magnitude and convert it into a salience distribution:

$$
s_i=\|\mathbf{r}_i\|_2,\qquad
w_i=\frac{\exp(s_i/\tau_s)}{\sum_{k=1}^{N}\exp(s_k/\tau_s)}.
\tag{9}
$$

Here \(\tau_s\) controls concentration. The within-image softmax ranks residual evidence without assuming cross-dataset calibration.

#### Salience-conditioned evidence of novelty

We re-aggregate residuals using both their original slot assignments and their surprise weights, yielding excited evidence that concentrates on coherent, unpredicted deviations:

$$
\mathbf{e}^{+}_j=\sum_{i=1}^{N}\tilde{A}_{ij}\,w_i\,\mathbf{r}_i,\qquad
\bar{\mathbf{e}}^{+}=\frac{1}{E}\sum_{j=1}^{E}\mathbf{e}^{+}_j.
\tag{10}
$$

Here \(\tilde{A}_{ij}\) preserves slot semantics, whereas \(w_i\) controls residual contribution. These equations separate **what recurring pattern a token belongs to** from **how unexpectedly it departs**.

#### Token refinement (broadcast of excited evidence)

Beyond global evidence, clinicians also refine a local mental map of the image. We broadcast excited evidence back to tokens with a lightweight readout:

$$
\mathbf{X}^{out}=\mathbf{X}+\phi(\mathbf{A}\mathbf{E}^{+}),\qquad
\mathbf{E}^{+}=[\mathbf{e}^{+}_1,\dots,\mathbf{e}^{+}_E]^{\top}\in\mathbb{R}^{E\times D},
\tag{11}
$$

where \(\phi(\cdot)\) is a small MLP applied row-wise.

### 4.3 Stage III: Complementary Accommodation

Discovery must integrate deviations into the global representation, yet an unconstrained update may distort known structure. Motivated by Complementary Learning Systems theory, SCAN constructs an additive update with reduced overlap with the dominant direction.

#### Novelty vector from excited evidence

We map excited evidence into a global novelty proposal via a simple first- and second-order readout:

$$
\Delta=\mathbf{W}_1\,\bar{\mathbf{e}}^{+}
+\mathbf{W}_2\left(\frac{1}{E}\sum_{j=1}^{E}\mathbf{e}^{+}_j\odot\mathbf{e}^{+}_j\right),
\tag{12}
$$

where \(\odot\) is the Hadamard product and \(\mathbf{W}_1,\mathbf{W}_2\in\mathbb{R}^{D\times D}\) are learnable. The first-order term retains the signed mean shift; the second records dimension-wise energy that may cancel in the mean. Together, they capture consistent and sparse-but-strong deviations without another attention block.

#### Complementarity

We reduce direct interference with the dominant content by removing the component of the proposal aligned with the anchor:

$$
\Delta_{\perp}=\Delta-(\Delta^{\top}\mathbf{u})\mathbf{u}.
\tag{13}
$$

We further summarize refined tokens by \(\mathbf{x}_{pool}=\mathrm{MeanPool}(\mathbf{X}^{out})\in\mathbb{R}^{D}\) and project this token evidence into the same complementary subspace:

$$
\mathbf{x}_{pool\perp}=\mathbf{x}_{pool}-(\mathbf{x}_{pool}^{\top}\mathbf{u})\mathbf{u}.
\tag{14}
$$

Projecting both branches into the same subspace prevents the pooled token summary from reintroducing the dominant component removed from the global proposal.

#### Non-parametric surprise gate

Rather than learning dataset-specific scalar calibrations, we define \(\gamma\) from three cognitively motivated signals: (i) surprise magnitude, (ii) knownness, and (iii) salience concentration via peakedness:

$$
S=\sum_{i=1}^{N}w_i s_i,\qquad
k=\cos(\mathbf{c},\bar{\mathbf{e}}),\qquad
\kappa=1-\frac{\mathrm{Ent}(\mathbf{w})}{\log N},
\qquad
\mathrm{Ent}(\mathbf{w})=-\sum_{i=1}^{N}w_i\log w_i.
\tag{15}
$$

Using EMA moments \((\mu_S,\sigma_S)\) and \((\mu_k,\sigma_k)\), we standardize the signals as

$$
\hat{S}=\frac{S-\mu_S}{\sigma_S},\qquad
\hat{k}=\frac{k-\mu_k}{\sigma_k}.
$$

The robust gate is

$$
\gamma=\sigma(\hat{S}-\hat{k})\cdot\mathrm{clip}(\kappa,0,1).
$$

The three factors are complementary: \(S\) measures deviation strength, \(k\) discounts samples already explained by dominant evidence, and \(\kappa\) suppresses diffuse residual activation. Accommodation is therefore selective in both residual magnitude and salience concentration.

#### Stable accommodation

We assimilate novelty as a complementary residual:

$$
\tilde{\mathbf{c}}=\mathbf{c}+\gamma\cdot\mathrm{LN}\left(\Delta_{\perp}+\mathbf{x}_{pool\perp}\right).
\tag{16}
$$

The identity path recovers the original representation when \(\gamma\) is small, and \(\mathrm{LN}(\cdot)\) controls update scale. This cannot guarantee complete independence from known features, but it biases accommodation away from the dominant direction and limits feature drift. The updated \(\tilde{\mathbf{c}}\) enters the unchanged GCD pipeline.

#### Connection to the GBD

The anchor has a setting-specific interpretation but a shared role:

- In Setting I, it summarizes head-class patterns, exposing underrepresented tail cues.
- In Setting II, it reflects recurring normal anatomy, retaining localized pathology in the residual.
- In Setting III, it captures parent-level morphology, while accommodation emphasizes subtype differences.

SCAN thereby targets the common suppression mechanism without setting-specific supervision.

## 5. Experiments

The experiments address three questions:

1. **Benchmark validity:** Does GBD capture the key open-world challenges in medical imaging, and can it differentiate methods across long-tail, anomaly-dominated, and taxonomy-aware settings?
2. **Effectiveness and generality:** Does SCAN improve generalized biomedicine discovery, and can it serve as a lightweight plug-and-play module that strengthens diverse host baselines without altering their original training protocols?
3. **Mechanism and practicality:** Which components drive the gains, how do they interact in the full framework, and how robust is SCAN under key hyperparameter choices and compute budgets?

### 5.1 Experimental Setup

#### Datasets

We evaluate GBD on four publicly available datasets spanning diverse biomedical imaging domains:

- **Gastrovision (endoscopy):** A multi-center GI endoscopy dataset spanning upper and lower GI tracts, with clinically meaningful finding types including anatomical landmarks/normal findings, pathological abnormalities, and procedure-related categories.
- **HistoSet-5×14 (histopathology):** A multi-organ H&E collection covering five organs (breast, colon, lung, oral cavity, and ovary), with a label space structured by organ identity and tissue state across 14 tissue classes.
- **MLL23 (microscopy):** Expert-annotated peripheral blood single-cell images whose labels follow hematopoietic structure with major groups and finer subdivisions into mature and immature cell types/stages.
- **DERM12345 (dermatology):** A dermatoscopic skin-lesion dataset with a taxonomic tree comprising 5 super-classes, 15 main classes, and 40 subclasses.

#### Evaluation protocol

For each dataset, models are trained on \(\mathcal{D}_l\cup\mathcal{D}_u\) without access to the ground-truth labels \(y_i^u\) in \(\mathcal{D}_u\). At test time, clustering accuracy between \(y_i^u\) and model predictions \(\hat y_i\) over \(\mathcal{D}_u\) is measured as:

$$
\mathrm{ACC}
=\max_{p\in\mathcal{P}(\mathcal{Y}_u)}
\frac{1}{|\mathcal{D}_u|}
\sum_{i=1}^{|\mathcal{D}_u|}
\mathbb{1}\{y_i^u=p(\hat y_i)\}.
\tag{17}
$$

Here \(\mathcal{P}(\mathcal{Y}_u)\) is the set of all permutations of class labels in the unlabeled label space \(\mathcal{Y}_u\). The maximization is computed via the Hungarian optimal assignment algorithm. ACC is reported on all unlabeled instances and separately on the known (Old) and novel (New) subsets. The Hungarian assignment is computed once over all classes in \(\mathcal{Y}_u\), and the same assignment is then used for the known/novel subset accuracies. Following the standard generalized discovery protocol, the number of novel classes \(K_{\mathrm{novel}}\) is assumed available for evaluation.

#### Implementation details

SCAN is implemented as a plug-and-play module attached to diverse GCD baselines. All methods use a ViT-B/16 backbone initialized with self-supervised DINO-V2 weights and are trained for 200 epochs. The default SCAN configuration uses \(E=16\) evidence slots. To evaluate generality, the original training hyperparameters and optimization schedules of each host baseline are largely retained without method-specific tuning. Following the original GCD configuration, 50% of samples from the known set are allocated to the unlabeled subset. We generally use \(\alpha=0.5\) in Setting I and \(\rho=0.5\) in Setting III.

#### Baselines

- **SimGCD:** A one-stage parametric classifier for GCD using debiased learning and entropy regularization.
- **LegoGCD:** A modular add-on targeting catastrophic forgetting in GCD by regularizing predictions to preserve known-class knowledge while improving novel discrimination.
- **SelEx:** A fine-grained GCD method that builds self-expertise via hierarchical pseudo-labeling and complementary supervised/unsupervised training signals.
- **SEAL:** A semantic-aware hierarchical framework that leverages naturally available hierarchies. Because it requires additional hierarchical labels, it is treated as an auxiliary performance reference rather than a directly comparable baseline.

### 5.2 Main Results

#### Results on Long-tail Rare Disease Discovery

| Method | Gastrovision All | Old | New | HistoSet All | Old | New | MLL23 All | Old | New | Derm12345 All | Old | New | Average All | Old | New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAL† | 56.8 | 51.2 | 60.7 | 81.2 | 87.1 | 77.9 | 65.4 | 67.6 | 54.1 | 56.5 | 61.0 | 10.6 | 65.0 | 66.7 | 50.8 |
| SelEx | 50.1 | 52.0 | 47.4 | 76.4 | 97.5 | 70.5 | 66.4 | 68.3 | 61.3 | 28.9 | 29.1 | 27.9 | 55.5 | 61.7 | 51.8 |
| SelEx + SCAN | **71.2** | **83.4** | 53.0 | 76.3 | **99.9** | 69.7 | **80.7** | **89.3** | 58.6 | **35.6** | 35.6 | **35.6** | **66.0** | **77.1** | **54.2** |
| SimGCD | 37.8 | 40.6 | 31.8 | 74.4 | 83.8 | 71.8 | 56.9 | 52.9 | 67.0 | 25.4 | 27.9 | 13.1 | 48.6 | 51.3 | 45.9 |
| SimGCD + SCAN | 39.7 | 43.3 | 31.9 | 80.0 | 82.1 | **79.4** | 64.3 | 63.1 | 67.2 | 25.9 | 28.2 | 14.2 | 52.5 | 54.2 | 48.1 |
| LegoGCD | 39.8 | 48.9 | 26.1 | 72.5 | 76.0 | 71.6 | 70.8 | 63.4 | **89.7** | 26.3 | 28.5 | 15.7 | 52.3 | 54.2 | 50.8 |
| LegoGCD + SCAN | 41.3 | 49.4 | 29.2 | 72.5 | 77.5 | 71.2 | 73.2 | 67.2 | 88.8 | 26.7 | 29.0 | 14.8 | 53.4 | 55.8 | 51.0 |

SCAN improves the average All accuracy of all three host baselines, yielding a notable +10.5-point gain on SelEx, although several subset-specific regressions remain. The effect is most pronounced on Gastrovision, where Old accuracy rises from 52.0% to 83.4% (+31.4 points), while on Derm12345 it boosts tail recognition from 27.9% to 35.6% (+7.7 points). SelEx+SCAN exceeds SEAL in several settings, although SEAL uses additional hierarchical supervision. These results support the motivation that predictive suppression attenuates dominant head-class patterns and surprise-evoked salience amplifies underrepresented tail cues.

#### Results on Normal-to-Abnormal Discovery

| Method | Gastrovision All | Old | New | HistoSet All | Old | New | MLL23 All | Old | New | Derm12345 All | Old | New | Average All | Old | New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAL† | 62.6 | 64.2 | 61.0 | 73.5 | 98.7 | 48.2 | 69.2 | 70.9 | 60.1 | 59.0 | 61.7 | 5.5 | 66.1 | 73.8 | 43.7 |
| SelEx | 50.3 | 51.9 | 41.7 | 64.3 | 97.7 | 47.5 | 58.5 | 58.0 | 59.8 | 29.3 | 28.2 | 40.7 | 50.6 | 58.9 | 47.4 |
| SelEx + SCAN | **70.1** | **86.0** | 46.3 | 68.6 | 97.2 | **60.7** | **87.1** | **89.1** | **81.7** | 33.6 | 33.3 | 36.4 | **64.8** | **76.4** | **56.3** |
| SimGCD | 39.7 | 42.9 | 25.6 | 61.1 | 94.1 | 44.5 | 57.9 | 55.9 | 63.4 | 26.6 | 27.2 | 19.9 | 46.3 | 55.0 | 38.3 |
| SimGCD + SCAN | 40.1 | 44.0 | 22.6 | 62.0 | 93.6 | 46.1 | 58.3 | 57.5 | 60.6 | 26.4 | 26.8 | 21.6 | 46.7 | 55.5 | 37.7 |
| LegoGCD | 49.3 | 53.7 | 24.9 | 62.0 | 97.5 | 44.2 | 68.5 | 72.1 | 58.6 | 28.5 | 30.8 | 5.5 | 52.1 | 63.5 | 33.3 |
| LegoGCD + SCAN | 50.5 | 55.5 | 23.1 | 63.9 | 97.4 | 47.1 | 69.4 | 73.7 | 57.7 | 29.0 | 29.7 | 21.6 | 53.2 | 64.1 | 37.4 |

With the strong SelEx baseline, SCAN improves average All accuracy by +14.2 points and abnormal New accuracy by +8.8 points. The impact is most pronounced on Gastrovision, where Old increases by 34.1 points, and on MLL23, where All and New rise by 28.6 and 21.9 points, respectively. Observed regressions are concentrated in particular host-baseline and subset combinations and may partly reflect using a fixed evidence-slot count \(E\) without per-domain tuning. Overall, the results are consistent with normal-anatomy suppression making subtle abnormalities more salient.

#### Results on Within-Taxonomy Discovery

| Method | Gastrovision All | Old | New | HistoSet All | Old | New | MLL23 All | Old | New | Derm12345 All | Old | New | Average All | Old | New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAL† | 54.9 | 51.3 | 64.3 | 92.9 | 95.7 | 89.3 | 70.6 | 83.3 | 47.2 | 27.7 | 62.4 | 16.4 | 61.5 | 73.2 | 54.3 |
| SelEx | 50.1 | 45.7 | 53.8 | 77.5 | 99.4 | 71.4 | 62.5 | 69.9 | 55.6 | 24.8 | 65.1 | 18.2 | 53.7 | 70.0 | 49.8 |
| SelEx + SCAN | **66.2** | **79.6** | 46.3 | 88.5 | 96.4 | 83.2 | **81.4** | **89.0** | **61.9** | **27.8** | **69.9** | **20.9** | **66.0** | **83.7** | 53.1 |
| SimGCD | 42.9 | 43.0 | 42.5 | 89.4 | 91.7 | 87.9 | 53.2 | 57.4 | 49.4 | 19.3 | 40.8 | 15.9 | 51.2 | 58.2 | 48.9 |
| SimGCD + SCAN | 42.0 | 42.4 | 39.4 | 89.1 | 90.5 | 88.2 | 54.0 | 58.4 | 49.9 | 17.8 | 36.8 | 14.7 | 50.7 | 57.0 | 48.1 |
| LegoGCD | 40.2 | 39.6 | 40.6 | 75.8 | 84.4 | 70.0 | 54.7 | 61.8 | 48.2 | 19.4 | 41.0 | 16.0 | 47.5 | 56.7 | 43.7 |
| LegoGCD + SCAN | 40.5 | 40.4 | 40.6 | 80.5 | 86.2 | 76.6 | 60.3 | 68.7 | 52.6 | 20.8 | 42.4 | 17.2 | 50.5 | 59.4 | 46.8 |

With SelEx, SCAN raises average All accuracy from 53.7% to 66.0% (+12.3 points), peaking on MLL23 (62.5% to 81.4%, +18.9 points) and Gastrovision (Old +33.8 points). SelEx+SCAN also exceeds SEAL on MLL23 New accuracy (61.9% vs. 47.2%) without extra fine-grained supervision. The observed drops are concentrated in particular datasets and host baselines and may partly reflect fixing \(E\) across domains to stress robustness.

### 5.3 Mechanism and Qualitative Analysis

**Figure 3. Training dynamics on Gastrovision (Setting I).** Known/novel accuracy and the evolution of the surprise signal \(S\) and knownness \(k\) for SelEx and SelEx+SCAN.

**Figure 4. Mechanism, qualitative, and efficiency analyses of SCAN.** Subfigures show evidence activations, UMAP embeddings, and computational overhead.

#### Adaptive Discovery Dynamics

Training dynamics on Gastrovision (Setting I) show that SCAN keeps Old accuracy comparable to SelEx, while New accuracy rises steadily and substantially exceeds the baseline. This improvement correlates with rising surprise signals, consistent with increased sensitivity to sparse, high-energy deviations. Around 40 epochs, SelEx shows a clear performance drop, whereas SCAN enters a second growth phase with fast accuracy gains. Meanwhile, the knownness indicator \(k\) slightly drops, corresponding to a less restrictive surprise gate for new samples. These trends suggest that SCAN preserves stable knowledge while reallocating representation capacity as training evolves.

#### Cross-Sample Consistency of Evidence Slots

Spatial activations on peripheral blood smear images suggest that evidence slots capture recurring cellular patterns. Evidence slot #3 often highlights high-contrast punctate textures around the perinuclear cytoplasm and salient cell boundaries, whereas evidence slot #9 tends to cover diffuse, low-intensity staining patterns in broader nuclear or cytoplasmic regions. These qualitative patterns suggest that the learned slots may provide reusable morphological primitives across samples.

#### Qualitative Analysis

UMAP visualizations on MLL23 (Setting I) show that SelEx yields dispersed clusters and noticeable overlap between new and old regions. With SCAN, within-class neighborhoods become tighter and margins among classes clearer. New cells appear more concentrated into coherent groups and more separated from old clusters, suggesting improved organization of subtle tail concepts.

#### Computational Overhead

SCAN adds a small number of parameters and several linear layers on top of the backbone and head. For SimGCD, the reported parameters increase from 92.10M to 99.68M (+7.6%), training time from 27.87s to 30.12s (+8%), and inference time from 10.61s to 11.32s (+6%). The measured increases are limited, suggesting that SCAN can be integrated into existing discovery pipelines with modest computational overhead.

### 5.4 Discussion

GBD is not intended as a leaderboard against text-supervised multimodal models (e.g., CLIP or LLMs), which assume a stable language anchor and reliable textual supervision. Instead, it targets upstream clinical **concept formation**: organizing unlabeled archives into coherent cohorts that experts can verify, name, merge, or split before integration into guidelines and curated datasets. This matters because emerging diseases and subtypes often lack consistent nomenclature across sites and time, while clinicians require cohort-level discovery, representative prototypes, and similar-case lists rather than one-off predictions.

## 6. Conclusion

We study generalized discovery in biomedical imaging under realistic clinical shifts. To support rigorous and comparable evaluation, we establish Generalized Biomedicine Discovery, a unified benchmark covering (i) long-tail rare disease discovery, (ii) normal-to-abnormal discovery, and (iii) within-taxonomy discovery across endoscopy, pathology, microscopy, and dermatology. Motivated by the central insight that dominant known patterns, especially normal anatomy, form a visual manifold that suppresses weak novelty cues, we propose SCAN, a cognition-inspired three-stage process of predictive suppression, surprise-evoked salience, and complementary accommodation. Extensive analyses and experiments support this insight, showing overall improvements in novel-disease discovery while revealing some dataset- and baseline-specific trade-offs in preserving established knowledge. GBD provides a clinically grounded, taxonomy-aware testbed that reframes biomedical imaging as a measurable concept-formation problem, enabling discovery systems to surface rare and emerging conditions and updateable subtypes with minimal reliance on exhaustive relabeling.

## Acknowledgments

The work described in this paper is supported by grants from HKU Startup Fund and HKU Seed Fund for Basic Research.
