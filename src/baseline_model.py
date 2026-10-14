import numpy as np
import pandas as pd


class BaselineModel:
    """
    Baseline friend recommendation model.
    
    For cold-start users, this model treats all contacts equally
    and ranks them randomly or by simple ordering.
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def rank_candidates(self, feature_df, user_col='user_id', method='random'):
        """
        Rank candidates for each user.
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            Feature dataframe with user_id, contact_id, and relevant columns
        user_col : str
            Column name for user ID
        method : str
            Ranking method: 'random' or 'contact_order'
        
        Returns:
        --------
        pandas.DataFrame: Ranked candidates with baseline_score column
        """
        ranked_df = feature_df.copy()
        
        if method == 'random':
            # Random ranking (treats all contacts equally)
            ranked_df['baseline_score'] = np.random.random(len(ranked_df))
        elif method == 'contact_order':
            # Simple ordering by contact ID
            ranked_df['baseline_score'] = ranked_df['contact_id'].astype(float)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by score descending within each user
        ranked_df = ranked_df.sort_values(
            [user_col, 'baseline_score'], 
            ascending=[True, False]
        )
        
        return ranked_df
    
    def get_top_k_recommendations(self, feature_df, user_col='user_id', k=10, method='random'):
        """
        Get top K recommendations for each user.
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            Feature dataframe
        user_col : str
            Column name for user ID
        k : int
            Number of recommendations to return per user
        method : str
            Ranking method: 'random' or 'contact_order'
        
        Returns:
        --------
        pandas.DataFrame: Top K recommendations for all users
        """
        ranked_df = self.rank_candidates(feature_df, user_col, method)
        
        top_k_list = []
        for user_id in ranked_df[user_col].unique():
            user_data = ranked_df[ranked_df[user_col] == user_id]
            top_k = user_data.head(k)
            top_k_list.append(top_k)
        
        if top_k_list:
            return pd.concat(top_k_list, ignore_index=True)
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import prepare_data
    
    # Load data
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    prepared_df = prepare_data(raw_df)
    
    # Run baseline
    baseline = BaselineModel()
    ranked_df = baseline.rank_candidates(prepared_df, method='random')
    
    print("\nBaseline ranking sample:")
    print(ranked_df.head(10))
    
    # Get top K recommendations
    top_k = baseline.get_top_k_recommendations(prepared_df, k=5)
    print(f"\nTop 5 recommendations: {len(top_k)} rows")
    print(top_k.head(10))