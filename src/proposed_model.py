import numpy as np
import pandas as pd
from relationship_score import add_relationship_scores, calculate_relationship_score


class ProposedModel:
    """
    Proposed friend recommendation model with temporal relationship score.
    
    For cold-start users, this model ranks contacts based on their
    temporal relationship-strength score derived from communication metadata.
    """
    
    def __init__(self, wf=0.40, wr=0.35, wi=0.25, k=10, tau_r=30, tau_i=7):
        self.wf = wf
        self.wr = wr
        self.wi = wi
        self.k = k
        self.tau_r = tau_r
        self.tau_i = tau_i
    
    def score_contacts(self, feature_df):
        """
        Calculate temporal relationship scores for all contacts.
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            Feature dataframe with n_interactions, median_interval_days,
            and last_interaction_days_ago columns
        
        Returns:
        --------
        pandas.DataFrame: Dataframe with relationship_score column added
        """
        return add_relationship_scores(
            feature_df,
            wf=self.wf, wr=self.wr, wi=self.wi,
            k=self.k, tau_r=self.tau_r, tau_i=self.tau_i
        )
    
    def rank_candidates(self, scored_df, user_col='user_id'):
        """
        Rank candidates by relationship score.
        
        Parameters:
        -----------
        scored_df : pandas.DataFrame
            Dataframe with relationship_score column
        user_col : str
            Column name for user ID
        
        Returns:
        --------
        pandas.DataFrame: Ranked candidates sorted by relationship score
        """
        ranked_df = scored_df.copy()
        ranked_df['proposed_score'] = ranked_df['relationship_score']
        
        # Sort by score descending within each user
        ranked_df = ranked_df.sort_values(
            [user_col, 'proposed_score'],
            ascending=[True, False]
        )
        
        return ranked_df
    
    def get_top_k_recommendations(self, scored_df, user_col='user_id', k=10):
        """
        Get top K recommendations for each user.
        
        Parameters:
        -----------
        scored_df : pandas.DataFrame
            Scored dataframe
        user_col : str
            Column name for user ID
        k : int
            Number of recommendations to return per user
        
        Returns:
        --------
        pandas.DataFrame: Top K recommendations for all users
        """
        ranked_df = self.rank_candidates(scored_df, user_col)
        
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
    
    # Run proposed model
    proposed = ProposedModel()
    scored_df = proposed.score_contacts(prepared_df)
    
    print("\nScored contacts sample:")
    print(scored_df.head(10))
    
    # Get top K recommendations
    top_k = proposed.get_top_k_recommendations(scored_df, k=5)
    print(f"\nTop 5 recommendations: {len(top_k)} rows")
    print(top_k.head(10))