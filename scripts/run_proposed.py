import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessing import prepare_data
from src.proposed_model import ProposedModel
from src.evaluation import evaluate_model


def main():
    print("Running Proposed Model Only...")
    
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    prepared_df = prepare_data(raw_df)
    
    proposed = ProposedModel(wf=0.40, wr=0.35, wi=0.25, k=10, tau_r=30, tau_i=7)
    scored_df = proposed.score_contacts(prepared_df)
    
    # Save scored data
    loader.save_scored_data(scored_df)
    
    ranked_df = proposed.rank_candidates(scored_df)
    
    results = evaluate_model(ranked_df, prepared_df, k_values=[5, 10, 20])
    
    print("\nProposed Results:")
    for k, metrics in results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")


if __name__ == "__main__":
    main()