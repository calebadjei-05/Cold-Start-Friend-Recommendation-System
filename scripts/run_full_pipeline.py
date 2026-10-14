import os
import sys
import json
import numpy as np
import pandas as pd

# Add the parent folder to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessing import prepare_data
from src.baseline_model import BaselineModel
from src.proposed_model import ProposedModel
from src.evaluation import (
    evaluate_model, compare_models, plot_comparison, 
    plot_score_distribution, plot_interaction_distribution,
    plot_confusion_matrix, save_metrics_to_json, create_error_analysis
)


def main():
    """This is the main function that runs everything."""
    
    print("=" * 70)
    print("Cold-Start Friend Recommendation System")
    print("Temporal Communication-Based Relationship Strength")
    print("=" * 70)
    
    # First, load the raw data
    print("\n[Step 1] Loading raw data...")
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    
    # Next, get the features and make labels
    print("\n[Step 2] Preparing data...")
    prepared_df = prepare_data(raw_df, n_interactions_threshold=5)
    
    # Now run the baseline model that treats everyone the same
    print("\n[Step 3] Running baseline model (treats contacts equally)...")
    baseline = BaselineModel(random_seed=42)
    baseline_ranked = baseline.rank_candidates(prepared_df, method='random')
    
    # Check how well the baseline did
    baseline_results = evaluate_model(baseline_ranked, prepared_df, k_values=[5, 10, 20])
    
    print("\nBaseline Results:")
    for k, metrics in baseline_results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")
    
    # Now run the proposed model that uses the time-based score
    print("\n[Step 4] Running proposed model (temporal relationship score)...")
    proposed = ProposedModel(wf=0.40, wr=0.35, wi=0.25, k=10, tau_r=30, tau_i=7)
    
    # Calculate the scores for each contact
    scored_df = proposed.score_contacts(prepared_df)
    
    # Save the scored data so we can look at it later
    loader.save_scored_data(scored_df)
    
    # Rank the candidates using the scores
    proposed_ranked = proposed.rank_candidates(scored_df)
    
    # Check how well the proposed model did
    proposed_results = evaluate_model(proposed_ranked, prepared_df, k_values=[5, 10, 20])
    
    print("\nProposed Results:")
    for k, metrics in proposed_results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")
    
    # Compare the two models side by side
    print("\n[Step 5] Comparing models...")
    comparison_df = compare_models(baseline_results, proposed_results, k_values=[5, 10, 20])
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Make some plots to show the results
    print("\n[Step 6] Generating plots...")
    plot_comparison(baseline_results, proposed_results, k_values=[5, 10, 20])
    plot_score_distribution(scored_df)
    plot_interaction_distribution(prepared_df)
    
    # Make confusion matrices to see where the models got things wrong
    print("\n[Step 7] Generating confusion matrices...")
    
    # For baseline, anything above 0.5 counts as relevant
    baseline_predictions = (baseline_ranked['baseline_score'] >= 0.5).astype(int)
    plot_confusion_matrix(
        baseline_ranked['relevant'].values,
        baseline_predictions,
        'Baseline Model Confusion Matrix',
        filename='confusion_matrix_baseline.png'
    )
    
    # For proposed, anything above 0.5 counts as relevant
    proposed_predictions = (proposed_ranked['relationship_score'] >= 0.5).astype(int)
    plot_confusion_matrix(
        proposed_ranked['relevant'].values,
        proposed_predictions,
        'Proposed Model Confusion Matrix',
        filename='confusion_matrix_proposed.png'
    )
    
    # Save all the numbers so we can use them later
    print("\n[Step 8] Saving metrics...")
    
    metrics_dict = {
        'model_comparison': {
            'baseline': baseline_results,
            'proposed': proposed_results
        },
        'parameters': {
            'wf': 0.40,
            'wr': 0.35,
            'wi': 0.25,
            'k': 10,
            'tau_r': 30,
            'tau_i': 7,
            'n_interactions_threshold': 5
        }
    }
    
    save_metrics_to_json(metrics_dict)
    
    # Look at the mistakes the model made
    print("\n[Step 9] Creating error analysis...")
    error_df = create_error_analysis(proposed_ranked, prepared_df)
    
    # Save the comparison table as a CSV file
    print("\n[Step 10] Saving comparison results...")
    comparison_df.to_csv('reports/comparison_results.csv', index=False)
    print("Comparison results saved to reports/comparison_results.csv")
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  Total user-contact pairs: {len(prepared_df)}")
    print(f"  Baseline Precision@10: {baseline_results[10]['precision']:.4f}")
    print(f"  Proposed Precision@10: {proposed_results[10]['precision']:.4f}")
    print(f"  Improvement: {(proposed_results[10]['precision'] - baseline_results[10]['precision'])*100:.2f}%")
    
    return baseline_results, proposed_results, comparison_df, scored_df


if __name__ == "__main__":
    baseline_results, proposed_results, comparison_df, scored_df = main()