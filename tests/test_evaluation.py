import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    evaluate_model
)


class TestEvaluation(unittest.TestCase):
    
    def setUp(self):
        """Create sample ranked data."""
        self.ranked_data = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2, 2],
            'contact_id': [2, 3, 4, 5, 3, 4, 5, 6],
            'relevant': [1, 1, 0, 0, 1, 0, 0, 0],
            'score': [0.9, 0.7, 0.5, 0.3, 0.8, 0.4, 0.3, 0.2]
        })
    
    def test_precision_at_k(self):
        """Test precision calculation."""
        precision = calculate_precision_at_k(self.ranked_data, k=2)
        # User 1: 2/2 relevant, User 2: 1/2 relevant
        # Average = (1.0 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(precision, 0.75, places=2)
    
    def test_recall_at_k(self):
        """Test recall calculation."""
        recall = calculate_recall_at_k(self.ranked_data, self.ranked_data, k=2)
        # User 1: 2/2 relevant retrieved, User 2: 1/1 relevant retrieved
        # Average = (1.0 + 1.0) / 2 = 1.0
        self.assertAlmostEqual(recall, 1.0, places=2)
    
    def test_hit_rate_at_k(self):
        """Test hit rate calculation."""
        hit_rate = calculate_hit_rate_at_k(self.ranked_data, k=2)
        # Both users have at least one relevant in top 2
        self.assertAlmostEqual(hit_rate, 1.0, places=2)
    
    def test_evaluate_model(self):
        """Test full evaluation."""
        results = evaluate_model(self.ranked_data, self.ranked_data, k_values=[2])
        
        self.assertIn(2, results)
        self.assertIn('precision', results[2])
        self.assertIn('recall', results[2])
        self.assertIn('hit_rate', results[2])


if __name__ == "__main__":
    unittest.main()