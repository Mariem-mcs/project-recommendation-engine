# Evaluation Metrics for Movie Recommendation System

## 1. Prediction Accuracy Metrics (Rating Prediction)

### RMSE (Root Mean Square Error)
**What it measures:** How far the predicted ratings are from actual ratings.

**Formula:** √(mean((predicted - actual)²))

**Interpretation:** Lower is better. Penalizes large errors more heavily.

**When to use:** When you care about getting ratings exactly right.

### MAE (Mean Absolute Error)
**What it measures:** Average absolute difference between predicted and actual ratings.

**Formula:** mean(|predicted - actual|)

**Interpretation:** Lower is better. Easier to interpret than RMSE.

**When to use:** When you care about the average error.

---

## 2. Ranking Quality Metrics (Recommendation Quality)

### Precision@K
**What it measures:** Fraction of recommended items that are relevant.

**Formula:** (Number of relevant items in top-K) / K

**Interpretation:** Higher is better. Measures how many of the recommendations are actually good.

**When to use:** When you care about not showing irrelevant items.

### Recall@K
**What it measures:** Fraction of all relevant items that appear in the top-K recommendations.

**Formula:** (Number of relevant items in top-K) / (Total number of relevant items)

**Interpretation:** Higher is better. Measures how many good items you're capturing.

**When to use:** When you care about not missing good items.

### F1@K
**What it measures:** Harmonic mean of Precision@K and Recall@K.

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Interpretation:** Higher is better. Balances precision and recall.

**When to use:** When you want a single metric that balances both.

### NDCG@K (Normalized Discounted Cumulative Gain)
**What it measures:** Ranking quality — positions of relevant items matter.

**Formula:** DCG / IDCG (where DCG = sum(relevance_i / log2(i+1)))

**Interpretation:** Higher is better (1.0 = perfect ranking). Rewards putting relevant items higher.

**When to use:** When the position of recommendations matters (which it usually does).

### Hit Rate@K
**What it measures:** Whether at least one relevant item appears in the top-K.

**Formula:** 1 if any relevant item in top-K, else 0

**Interpretation:** Higher is better. Measures if the user got at least one good recommendation.

**When to use:** When you care about user satisfaction (getting something relevant).

### MRR@K (Mean Reciprocal Rank)
**What it measures:** The position of the first relevant item.

**Formula:** 1 / position_of_first_relevant_item

**Interpretation:** Higher is better. Measures how quickly users find something relevant.

**When to use:** When ranking order matters.

---

## 3. Which Metrics to Use

| Metric | Why We Use It |
| **RMSE** | Standard for rating prediction accuracy |
| **Precision@10** | Standard for recommendation quality |
| **Recall@10** | Measures how many good items we capture |
| **NDCG@10** | Measures ranking quality (position matters) |
| **Hit Rate@10** | Measures user satisfaction |
