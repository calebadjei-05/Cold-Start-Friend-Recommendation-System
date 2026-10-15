# Cold-Start Friend Recommendation System

A research project that tests whether a time-based relationship score can make friend recommendations better for new users.

---

## What This Project Is About

Large friend recommendation systems usually work in two steps: first they find candidates, then they rank them. But just because someone is reachable does not mean they are actually a strong or current friend. This is a big problem for cold-start users, who just joined the platform and have almost no history to learn from.

This project builds two models and compares them:

- **Baseline Model:** Treats all contacts the same and ranks them randomly.
- **Proposed Model:** Gives each contact a time-based relationship score using frequency, recency, and inter-event time.

The goal is to see if the time-based score helps recommend better friends for cold-start users.

---

## Dataset

**CollegeMsg dataset from Stanford SNAP**

- Link: https://snap.stanford.edu/data/CollegeMsg.html
- 59,835 interactions between 1,350 users
- Contains sender ID, receiver ID, and timestamp
- No message content, no names, fully anonymous

Place the raw file in the `data/` folder as `raw_contact_data.csv`.

---

## Folder Structure

```
cold_start_friend_recommendation/
│
├── .github/workflows/          # GitHub Actions workflow
├── data/                       # Raw and processed data
│   ├── raw_contact_data.csv
│   └── temporal_scored_contacts.csv
├── notebooks/                  # Jupyter notebook for full analysis
│   └── full_analysis_pipeline.ipynb
├── reports/                    # Results and figures
│   ├── figures/
│   ├── error_analysis.csv
│   └── metrics.json
├── scripts/                    # Scripts to run the project
│   ├── run_baseline.py
│   ├── run_full_pipeline.py
│   └── run_proposed.py
├── src/                        # Python source code
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── proposed_model.py
│   └── relationship_score.py
├── tests/                      # Unit tests
│   ├── test_evaluation.py
│   ├── test_preprocessing.py
│   └── test_relationship_score.py
├── .gitignore
├── ai_use_disclosure.md
├── data_card.md
├── ethics_statement.md
├── failure_log.md
├── literature_matrix.md
├── model_card.md
├── PROJECT_BRIEF.md
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### Step 1: Set Up Virtual Environment

Open PowerShell or Command Prompt in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Requirements

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the Full Pipeline

```powershell
python scripts/run_full_pipeline.py
```

This will:

- Load the raw data
- Extract time-based features
- Calculate relationship scores
- Run both models
- Evaluate results
- Save figures and reports

### Step 4: Run Tests

```powershell
python -m pytest tests/ -v -s
```

### Step 5: Open the Notebook

```powershell
jupyter notebook notebooks/full_analysis_pipeline.ipynb
```

---

## Results

| Model | Precision@10 | Recall@10 | Hit Rate@10 |
| :--- | :--- | :--- | :--- |
| Baseline | 0.0743 | 0.2850 | 0.4089 |
| Proposed | 0.1707 | 0.4368 | 0.4615 |

The proposed model did about 130% better on Precision@10.

---

## Project Files Explained

| File | What it is |
| :--- | :--- |
| `README.md` | This file. Explains the project. |
| `PROJECT_BRIEF.md` | Short summary of the hackathon brief. |
| `data_card.md` | Info about the dataset. |
| `model_card.md` | Info about the models and scores. |
| `ethics_statement.md` | Privacy and safety notes. |
| `ai_use_disclosure.md` | Which AI tools were used and how. |
| `failure_log.md` | Mistakes and how they were fixed. |
| `literature_matrix.md` | Summary of papers read. |
| `requirements.txt` | Python packages needed. |

---

## Tech Stack

- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- pytest
- Jupyter Notebook

---

## Notes

- No private data is used.
- No message content is read.
- User IDs are anonymous numbers.
- The dataset is public and free to use.

---

## Author

Adjei Caleb Yaw Kakra
Student ID: 22040866
GitHub: calebadjei-05
