"""
Preprocessing module for the cold-start friend recommendation system.
Extracts temporal features and creates relevance labels.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def convert_timestamp_to_datetime(timestamp_series):
    """
    Convert Unix timestamp to datetime.
    
    Parameters:
    -----------
    timestamp_series : pandas.Series
        Series of Unix timestamps
    
    Returns:
    --------
    pandas.Series: Datetime objects
    """
    return pd.to_datetime(timestamp_series, unit='s')


def extract_temporal_features(raw_df, user_col='user_id', contact_col='contact_id', 
                              timestamp_col='timestamp'):
    """
    Extract temporal features for each user-contact pair.
    
    Parameters:
    -----------
    raw_df : pandas.DataFrame
        Raw interaction data
    user_col : str
        Column name for user ID
    contact_col : str
        Column name for contact ID
    timestamp_col : str
        Column name for timestamp
    
    Returns:
    --------
    pandas.DataFrame: Feature dataframe with one row per user-contact pair
    """
    # Convert timestamp to datetime
    raw_df = raw_df.copy()
    raw_df['datetime'] = convert_timestamp_to_datetime(raw_df[timestamp_col])
    
    # Get reference date (most recent timestamp in entire dataset)
    reference_date = raw_df['datetime'].max()
    
    # Group by user and contact
    grouped = raw_df.groupby([user_col, contact_col])
    
    features = []
    
    for (user, contact), group in grouped:
        interactions = group['datetime'].sort_values()
        
        n_interactions = len(interactions)
        
        # Calculate inter-event times (gaps between consecutive interactions)
        if n_interactions > 1:
            inter_event_times = interactions.diff().dt.total_seconds() / 86400.0  # Convert to days
            inter_event_times = inter_event_times.dropna()
            median_interval = inter_event_times.median()
        else:
            median_interval = 365.0  # Default for single interaction
        
        # Calculate recency (days since last interaction)
        last_interaction = interactions.max()
        days_since_last = (reference_date - last_interaction).total_seconds() / 86400.0
        
        features.append({
            'user_id': user,
            'contact_id': contact,
            'n_interactions': n_interactions,
            'median_interval_days': max(median_interval, 0.01),  # Avoid zero division
            'last_interaction_days_ago': max(days_since_last, 0),
        })
    
    feature_df = pd.DataFrame(features)
    print(f"Extracted features for {len(feature_df)} user-contact pairs")
    
    return feature_df


def create_relevance_labels(feature_df, n_interactions_threshold=5):
    """
    Create binary relevance labels based on interaction count.
    
    A contact is considered 'relevant' if the user has interacted with them
    more than the threshold number of times.
    
    Parameters:
    -----------
    feature_df : pandas.DataFrame
        Feature dataframe
    n_interactions_threshold : int
        Minimum number of interactions for a contact to be relevant
    
    Returns:
    --------
    pandas.DataFrame: Feature dataframe with relevance label added
    """
    labeled_df = feature_df.copy()
    labeled_df['relevant'] = (labeled_df['n_interactions'] >= n_interactions_threshold).astype(int)
    
    n_relevant = labeled_df['relevant'].sum()
    n_total = len(labeled_df)
    
    print(f"Created relevance labels:")
    print(f"  Relevant contacts: {n_relevant} ({n_relevant/n_total*100:.1f}%)")
    print(f"  Non-relevant contacts: {n_total - n_relevant} ({(n_total - n_relevant)/n_total*100:.1f}%)")
    
    return labeled_df


def prepare_data(raw_df, n_interactions_threshold=5):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    raw_df : pandas.DataFrame
        Raw interaction data
    n_interactions_threshold : int
        Threshold for relevance label
    
    Returns:
    --------
    pandas.DataFrame: Prepared data with features and labels
    """
    # Extract temporal features
    feature_df = extract_temporal_features(raw_df)
    
    # Create relevance labels
    labeled_df = create_relevance_labels(feature_df, n_interactions_threshold)
    
    return labeled_df


if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    
    prepared_df = prepare_data(raw_df)
    
    print("\nPrepared data:")
    print(prepared_df.head(10))
    
    print("\nFeature statistics:")
    print(prepared_df.describe())