# Model Card

## Required details

This model recommends friends on a social network. Given a user and a list of possible contacts, it ranks the contacts by how likely they are to form a meaningful relationship with that user.

The dataset contains communication records between users, including when each interaction happened. These timestamps are used to build features that measure relationship strength over time.

The proposed model ranks contacts using a temporal relationship-strength score. The baseline ranks contacts randomly. Both were tested on the same dataset using two metric types: classification metrics at a 0.5 threshold, and ranking metrics (Precision, Recall, Hit Rate) at K = 5, 10, and 20.

The results from the notebook pipeline are shown below.

### Classification Metrics (0.5 score threshold)

**Baseline**
- Accuracy: 0.5016
- Precision: 0.1478
- Recall: 0.5029
- F1: 0.2285

**Proposed**
- Accuracy: 0.9018
- Precision: 0.9490
- Recall: 0.3498
- F1: 0.5112

### Ranking Metrics (Precision / Recall / Hit Rate @ K)

**Baseline**
- K=5 — Precision: 0.0861, Recall: 0.1883, Hit Rate: 0.3081
- K=10 — Precision: 0.0743, Recall: 0.2850, Hit Rate: 0.4089
- K=20 — Precision: 0.0585, Recall: 0.3713, Hit Rate: 0.4526

**Proposed**
- K=5 — Precision: 0.2538, Recall: 0.3961, Hit Rate: 0.4607
- K=10 — Precision: 0.1707, Recall: 0.4368, Hit Rate: 0.4615
- K=20 — Precision: 0.1007, Recall: 0.4548, Hit Rate: 0.4615

### Limitations

The baseline ranks contacts randomly, making it a weak point of comparison.

The definition of relevance is narrow and may not capture every type of meaningful relationship.

No statistical significance test was performed, so the improvements cannot be confirmed as statistically significant.

The dataset may reflect a specific platform or user group, limiting generalisability to other platforms or populations.

Accuracy requires careful interpretation: 85.3% of user pairs are non-relevant, so a model that always predicts "non-relevant" would already achieve about 0.853 accuracy.

### Verification

-Results were produced by the notebook pipeline after running the model on the dataset.
-Both models were evaluated on the same  test set of the dataset.
-Classification metrics were computed at a score threshold of 0.5.
-Results match the notebook outputs and the values reported in Section 6.1 of the reserach document.
Results saved in:`notebooks/full_analysis_pipeline.ipynb`