import os
import pandas as pd
import numpy as np
from datetime import datetime


class DataLoader:
    """Load and manage contact interaction data."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_file = os.path.join(data_dir, 'raw_contact_data.csv')
        self.scored_file = os.path.join(data_dir, 'temporal_scored_contacts.csv')
    
    def load_raw_data(self):
        """
        Load the raw contact data.
        
        Returns:
        --------
        pandas.DataFrame: Raw interaction data with columns [user_id, contact_id, timestamp]
        """
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(
                f"Raw data file not found at {self.raw_file}. "
                "Please ensure raw_contact_data.csv is in the data folder."
            )
        
        df = pd.read_csv(self.raw_file)
        print(f"Loaded raw data: {len(df)} interactions")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Unique contacts: {df['contact_id'].nunique()}")
        
        return df
    
    def load_scored_data(self):
        """
        Load the temporal scored data if it exists.
        
        Returns:
        --------
        pandas.DataFrame or None: Scored data if file exists, None otherwise
        """
        if os.path.exists(self.scored_file):
            df = pd.read_csv(self.scored_file)
            print(f"Loaded scored data: {len(df)} rows")
            return df
        else:
            print("Scored data file not found. Run the proposed model first.")
            return None
    
    def save_scored_data(self, scored_df):
        """
        Save the temporal scored data to CSV.
        
        Parameters:
        -----------
        scored_df : pandas.DataFrame
            Dataframe with temporal score columns added
        """
        scored_df.to_csv(self.scored_file, index=False)
        print(f"Saved scored data to {self.scored_file}")
        print(f"Total rows: {len(scored_df)}")
        print(f"Columns: {list(scored_df.columns)}")
    
    def get_data_summary(self, df):
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'unique_contacts': df['contact_id'].nunique() if 'contact_id' in df.columns else 0,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'timestamp_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None
        }
        return summary


if __name__ == "__main__":
    loader = DataLoader()
    
    # Load raw data
    raw_df = loader.load_raw_data()
    
    # Get summary
    summary = loader.get_data_summary(raw_df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")