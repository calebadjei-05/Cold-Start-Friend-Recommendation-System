import numpy as np
import pandas as pd


def calculate_frequency_score(n_interactions, k=10):
    # F = 1 - e^(-n/k)
    return 1 - np.exp(-np.array(n_interactions) / k)


def calculate_recency_score(days_since_last_interaction, tau_r=30):
    # R = e^(-d/tau_r)
    return np.exp(-np.array(days_since_last_interaction) / tau_r)


def calculate_interevent_score(median_interval_days, tau_i=7):
    # I = e^(-m/tau_i)
    return np.exp(-np.array(median_interval_days) / tau_i)


def calculate_relationship_score(n_interactions, days_since_last_interaction,
                                 median_interval_days, wf=0.40, wr=0.35, wi=0.25,
                                 k=10, tau_r=30, tau_i=7):
    # Score = (F × Wf) + (R × Wr) + (I × Wi)
    assert abs(wf + wr + wi - 1.0) < 1e-6, "Weights must sum to 1"

    F = calculate_frequency_score(n_interactions, k)
    R = calculate_recency_score(days_since_last_interaction, tau_r)
    I = calculate_interevent_score(median_interval_days, tau_i)

    score = (wf * F) + (wr * R) + (wi * I)

    return score


def interpret_score(score):
    if score < 0.25:
        return "Weak or No Relationship Signal"
    elif score < 0.50:
        return "Moderate Relationship Signal"
    elif score < 0.75:
        return "Strong Relationship Signal"
    else:
        return "Very Strong Relationship Signal"


def add_relationship_scores(feature_df, wf=0.40, wr=0.35, wi=0.25,
                            k=10, tau_r=30, tau_i=7):
    scored_df = feature_df.copy()

    scores = calculate_relationship_score(
        scored_df['n_interactions'].values,
        scored_df['last_interaction_days_ago'].values,
        scored_df['median_interval_days'].values,
        wf=wf, wr=wr, wi=wi, k=k, tau_r=tau_r, tau_i=tau_i
    )

    scored_df['relationship_score'] = scores
    scored_df['score_interpretation'] = [interpret_score(s) for s in scores]

    scored_df['frequency_score'] = calculate_frequency_score(
        scored_df['n_interactions'].values, k
    )
    scored_df['recency_score'] = calculate_recency_score(
        scored_df['last_interaction_days_ago'].values, tau_r
    )
    scored_df['interevent_score'] = calculate_interevent_score(
        scored_df['median_interval_days'].values, tau_i
    )

    print(f"Added relationship scores to {len(scored_df)} rows")

    return scored_df


if __name__ == "__main__":
    test_cases = [
        {'n': 0, 'd': 365, 'm': 90, 'label': 'No interaction'},
        {'n': 2, 'd': 200, 'm': 100, 'label': 'Rare, old interaction'},
        {'n': 5, 'd': 90, 'm': 30, 'label': 'Occasional interaction'},
        {'n': 20, 'd': 30, 'm': 10, 'label': 'Frequent interaction'},
        {'n': 60, 'd': 7, 'm': 2, 'label': 'Very frequent, recent'},
        {'n': 150, 'd': 1, 'm': 0.5, 'label': 'Constant communication'},
    ]

    print("Temporal Relationship Score Test Cases:")
    print("-" * 60)

    for case in test_cases:
        score = calculate_relationship_score(
            case['n'], case['d'], case['m']
        )
        print(f"{case['label']}:")
        print(f"  n={case['n']}, d={case['d']}, m={case['m']}")
        print(f"  Score = {score:.4f} → {interpret_score(score)}")
        print()