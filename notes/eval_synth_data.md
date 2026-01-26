## Standard diagnostics for synthetic data quality

**1. KS Mean (Kolmogorov–Smirnov Mean)**  
KS statistic measures how different two distributions are. For a feature \(X\):  
$$
\text{KS}(X_{\text{real}}, X_{\text{synthetic}}) = \max_x \left|F_{\text{real}}(x) - F_{\text{synthetic}}(x)\right|
$$
where \(F(x)\) is the cumulative distribution function (CDF).  
Value range: 0 – 1  

**KS Mean:**  
$$
\text{KS Mean} = \frac{1}{d} \sum_{i=1}^{d} \text{KS}(X_i^{\text{real}}, X_i^{\text{synthetic}})
$$

**Interpretation:**  
- 0.00 – 0.05 → very realistic marginals  
- 0.05 – 0.10 → acceptable  
- 0.10 → synthetic data deviates noticeably  

**Why it matters in fraud:**  
- SMOTE often looks good in KS (smooth interpolation)  
- GANs can look worse if poorly trained (mode collapse)  

**Note:** KS only checks marginal distributions — it does not check relationships between features.

2. **Correlation Gap (Dependency Preservation)**  
    **What it is:** Measures how well the relationships between features are preserved.  

    **Procedure:**  
    - Compute correlation matrix on real data \(C_{\text{real}} \in \mathbb{R}^{d \times d}\)  
    - Compute correlation matrix on synthetic data \(C_{\text{syn}}\)  
    - Compute matrix distance, e.g., Frobenius norm:  
      \[
      \text{Correlation Gap} = \|C_{\text{real}} - C_{\text{syn}}\|_F
      \]

    **Interpretation:**  
    - Small value → feature dependencies preserved  
    - Large value → synthetic data breaks joint structure  

    **Why it matters in fraud:**  
    Fraud is interaction-driven (amount × time, device × velocity, claim amount × policy age).  
    - SMOTE often fails here  
    - GAN/CTGAN should outperform SMOTE if implemented well

3. **TSTR Score (Train on Synthetic, Test on Real)**  
    **What it is:** Task-based realism test.  

    **Procedure:**  
    - Train a classifier only on synthetic data  
    - Test it on real test data  
    - Measure PR-AUC / Recall / F1  

    **Interpretation:**  
    - High TSTR → synthetic data captures decision-relevant patterns  
    - Low TSTR → synthetic data looks “real” but is useless for learning fraud  

    **Why it matters more than KS:**  
    KS can be low while TSTR is terrible. TSTR directly answers: “Can synthetic fraud replace real fraud for learning?”  
    Reviewers trust TSTR more than distribution metrics.

4. **Duplicate Rate (Mode Collapse Indicator)**  
    **What it is:** Measures how often the generator produces exact or near-duplicates.  

    **Formula:**  
    \[
    \text{Duplicate Rate} = \frac{\text{Number of duplicate (or near-duplicate) rows}}{\text{Total synthetic rows}}
    \]

    **Why it matters:**  
    High duplicate rate = mode collapse (memorization instead of generalization).  

    **Typical ranges:**  
    - < 1% → good diversity  
    - 1–5% → acceptable  
    - 5% → problematic  
    - 10% → unusable synthetic data  

    - SMOTE often has very high duplicate/near-duplicate rate  
    - Good GANs reduce this significantly  

## How these metrics complement each other

| Metric           | Checks               | What it misses           |
|------------------|----------------------|--------------------------|
| KS Mean          | Marginal realism     | Feature relationships    |
| Correlation Gap  | Dependencies         | Decision usefulness      |
| TSTR             | Task realism         | Distribution fidelity    |
| Duplicate Rate   | Diversity            | Semantic correctness     |

👉 No single metric is sufficient alone  
👉 Together, they form a defensible evaluation suite