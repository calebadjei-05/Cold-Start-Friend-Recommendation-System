import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    extract_temporal_features,
    create_relevance_labels,
    prepare_data
)


class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample raw data."""
        self.raw_data = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2],
            'contact_id': [2, 2, 3, 3, 4],
            'timestamp': [1082040961, 1082414391, 1082155839, 1082439619, 1082439756]
        })
    
    def test_extract_temporal_features(self):
        """Test feature extraction."""
        features = extract_temporal_features(self.raw_data)
        
        self.assertIn('n_interactions', features.columns)
        self.assertIn('median_interval_days', features.columns)
        self.assertIn('last_interaction_days_ago', features.columns)
        
        # User 1 has 2 interactions with contact 2
        user1_contact2 = features[(features['user_id'] == 1) & (features['contact_id'] == 2)]
        self.assertEqual(user1_contact2['n_interactions'].values[0], 2)
    
    def test_create_relevance_labels(self):
        """Test label creation."""
        features = extract_temporal_features(self.raw_data)
        labeled = create_relevance_labels(features, n_interactions_threshold=2)
        
        self.assertIn('relevant', labeled.columns)
        self.assertEqual(labeled['relevant'].dtype, np.int64)
    
    def test_prepare_data(self):
        """Test full preprocessing pipeline."""
        prepared = prepare_data(self.raw_data, n_interactions_threshold=2)
        
        self.assertIn('n_interactions', prepared.columns)
        self.assertIn('relevant', prepared.columns)
        self.assertGreater(len(prepared), 0)


if __name__ == "__main__":
    unittest.main()