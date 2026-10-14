# DCIT 316 Mid-Semester Hackathon 2026 — Project Brief

## Title

**Cold-Start Friend Recommendation Using Temporal Communication-Based Relationship Strength**

 

## Purpose

This project is about testing whether a time-based relationship score can help me recommend friends better for new users. I use a public dataset, build a proper pipeline, compare two ways of ranking, check the results with standard metrics, and document all ethical considerations.

 

## Research Problem


Big friend-recommendation systems usually work in two steps: first they find candidates, then they rank them. Jun Yu (2024) explains how this works. They pull candidates from friend graphs and contact books, then filter them down to about 10,000 people before ranking.

However, just because someone is reachable does not mean they are actually a strong or current friend. This is a major issue for cold-start users. A cold-start user is someone who just joined the platform and has almost no history. They have no friends yet, no interactions, and no activity for the system to learn from. So the usual ways of finding candidates, like Friend-of-Friend, do not work for them because there are no friends to start from. The system has to rely on contact books or other outside data just to find anyone at all. Even then, it cannot tell which contact is actually close and which one is just an old number saved years ago. That is the gap this project is trying to fill..

 

## Research Question

“Can a temporal communication-based relationship-strength score can improve the relevance of friend recommendations for cold-start users.”

 

## Data Source

**Dataset:** CollegeMsg dataset from Stanford SNAP  
**Link:** [https://snap.stanford.edu/data/CollegeMsg.html](https://snap.stanford.edu/data/CollegeMsg.html)

**What it is:**  
This dataset has 59,835 messages between 1,350 users at a university. Each row has a sender ID, a receiver ID, and a timestamp. It is public, anonymous, and contains no message content. I use it only for this project.

**How I got it:**  
I downloaded it for free from Stanford SNAP.

**Ethics:**

- I do not collect private data.
- I do not read any messages.
- User IDs are just numbers.
- I do not try to identify anyone.
- I use the dataset only for learning and research.

 

## Method and Pipeline

I build two models and compare them on the same data.

### Baseline Model

The baseline treats all contacts the same and ranks them randomly. This is like a system with no time information.

### Proposed Model

The proposed model gives each contact a time-based relationship score using three components:

1. **Frequency:** how often two people talk.
2. **Recency:** how recently they last talked.
3. **Inter-event time:** how far apart their conversations usually are.

The score ranges from 0 to 1 and is used to rank contacts. I also create labels for evaluation using a rule based on how many times users interacted.

### Pipeline Steps

- Setup and imports.
- Data loading.
- Exploratory Data Analysis.
- Data preprocessing.
- Temporal relationship score calculation.
- Baseline model.
- Proposed model.
- Model comparison.
- Evaluation metrics.
- Confusion matrices.
- Error analysis.
- Comprehensive visualizations.
- Results summary and conclusion.

 

## Evaluation

| Model | Precision@10 | Recall@10 | Hit Rate@10 |
| :  | : : | : : | : : |
| Baseline | 0.0743 | 0.2850 | 0.4089 |
| Proposed | 0.1707 | 0.4368 | 0.4615 |

The proposed model performed about **130% better** on Precision@10 compared with the baseline.

 

## Error Analysis

I compared the model’s predictions with the actual labels. Most mistakes occurred when relevant contacts had low time-based scores, usually because the interactions were old or there were not many of them. I saved these cases in `reports/error_analysis.csv`.

 

## Limitations

- The baseline ranks contacts randomly, making it a weak point of comparison.

- The definition of relevance is narrow and may not capture every type of meaningful relationship.

- No statistical significance test was performed, so the improvements cannot be confirmed as statistically significant.

- The dataset may reflect a specific platform or user group, limiting generalisability to other platforms or populations.

- Accuracy requires careful interpretation: 85.3% of user pairs are non-relevant, so a model that always predicts "non-relevant" would already achieve about 0.853 accuracy.

 

## Ethical Considerations

I did not collect private data. I did not read any messages. User IDs are anonymous. The dataset is public and appropriate for research. I did not attempt to identify anyone. I used only timestamps, which follows the privacy-conscious approach discussed by Rosenfeld et al. (2018).

 

## Required Repository Evidence

| File | What it is for |
| :  | :  |
| `README.md` | Explains the project and how to run it |
| `data_card.md` | Information about the dataset |
| `ethics_statement.md` | Privacy and safety considerations |
| `model_card.md` | Information about the models and their scores |
| `ai_use_disclosure.md` | Which AI tools I used and how |
| `failure_log.md` | Errors I encountered and how I fixed them |
| `literature_matrix.md` | Summary of papers I read |
| `notebooks/` | Jupyter notebooks for the model pipeline |
| `src/` | Python files for data, scoring, and models |
| `reports/figures/` | All charts and plots |
| `research document/` | Presentation slides |



 

## Academic Integrity

I used AI tools for brainstorming, fixing bugs, and explaining concepts. All AI use is documented in `ai_use_disclosure.md`.

I can explain my code, data choices, model choices, papers, errors, and results.

 

## Expected Contribution

A time-based relationship score that can be added to existing friend-recommendation pipelines to help new users receive better recommendations.

This feature works alongside existing methods, not instead of them.