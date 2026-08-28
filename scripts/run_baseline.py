import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessing import prepare_data
from src.baseline_model import BaselineModel
from src.evaluation import evaluate_model


def main():
    print("Running Baseline Model Only...")
    
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    prepared_df = prepare_data(raw_df)
    
    baseline = BaselineModel(random_seed=42)
    ranked_df = baseline.rank_candidates(prepared_df, method='random')
    
    results = evaluate_model(ranked_df, prepared_df, k_values=[5, 10, 20])
    
    print("\nBaseline Results:")
    for k, metrics in results.items():
        print(f"  K={k}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Hit Rate={metrics['hit_rate']:.4f}")


if __name__ == "__main__":
    main()