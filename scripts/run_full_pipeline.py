import os
import sys
import json
import numpy as np
import pandas as pd

# Add parent directory to path
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
    """Main execution function."""
    
    print("=" * 70)
    print("Cold-Start Friend Recommendation System")
    print("Temporal Communication-Based Relationship Strength")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading raw data...")
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    
    # Step 2: Prepare data (extract features and create labels)
    print("\n[Step 2] Preparing data...")
    prepared_df = prepare_data(raw_df, n_interactions_threshold=5)
    
    # Step 3: Run baseline model
    print("\n[Step 3] Running baseline model (treats contacts equally)...")
    baseline = BaselineModel(random_seed=42)
    baseline_ranked = baseline.rank_candidates(prepared_df, method='random')
    
    # Evaluate baseline
    baseline_results = evaluate_model(baseline_ranked, prepared_df, k_values=[5, 10, 20])
    
    print("\nBaseline Results:")
    for k, metrics in baseline_results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")
    
    # Step 4: Run proposed model
    print("\n[Step 4] Running proposed model (temporal relationship score)...")
    proposed = ProposedModel(wf=0.40, wr=0.35, wi=0.25, k=10, tau_r=30, tau_i=7)
    
    # Score contacts
    scored_df = proposed.score_contacts(prepared_df)
    
    # Save scored data
    loader.save_scored_data(scored_df)
    
    # Rank candidates
    proposed_ranked = proposed.rank_candidates(scored_df)
    
    # Evaluate proposed
    proposed_results = evaluate_model(proposed_ranked, prepared_df, k_values=[5, 10, 20])
    
    print("\nProposed Results:")
    for k, metrics in proposed_results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")
    
    # Step 5: Compare models
    print("\n[Step 5] Comparing models...")
    comparison_df = compare_models(baseline_results, proposed_results, k_values=[5, 10, 20])
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Step 6: Generate plots
    print("\n[Step 6] Generating plots...")
    plot_comparison(baseline_results, proposed_results, k_values=[5, 10, 20])
    plot_score_distribution(scored_df)
    plot_interaction_distribution(prepared_df)
    
    # Step 7: Confusion matrices
    print("\n[Step 7] Generating confusion matrices...")
    
    # For baseline (random ranking, predict all as non-relevant at threshold 0.5)
    baseline_predictions = (baseline_ranked['baseline_score'] >= 0.5).astype(int)
    plot_confusion_matrix(
        baseline_ranked['relevant'].values,
        baseline_predictions,
        'Baseline Model Confusion Matrix',
        filename='confusion_matrix_baseline.png'
    )
    
    # For proposed (relationship score >= 0.5 as relevant)
    proposed_predictions = (proposed_ranked['relationship_score'] >= 0.5).astype(int)
    plot_confusion_matrix(
        proposed_ranked['relevant'].values,
        proposed_predictions,
        'Proposed Model Confusion Matrix',
        filename='confusion_matrix_proposed.png'
    )
    
    # Step 8: Save metrics
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
    
    # Step 9: Error analysis
    print("\n[Step 9] Creating error analysis...")
    error_df = create_error_analysis(proposed_ranked, prepared_df)
    
    # Step 10: Save comparison results
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