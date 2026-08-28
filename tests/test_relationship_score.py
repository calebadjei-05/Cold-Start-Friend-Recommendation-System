import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.relationship_score import (
    calculate_frequency_score,
    calculate_recency_score,
    calculate_interevent_score,
    calculate_relationship_score,
    interpret_score
)


class TestRelationshipScore(unittest.TestCase):
    
    def test_frequency_score_range(self):
        """Frequency score should be between 0 and 1."""
        scores = calculate_frequency_score([0, 5, 10, 50, 100])
        for score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_frequency_score_monotonic(self):
        """More interactions should give higher frequency score."""
        scores = calculate_frequency_score([1, 10, 100])
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
    
    def test_recency_score_range(self):
        """Recency score should be between 0 and 1."""
        scores = calculate_recency_score([0, 10, 50, 100])
        for score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_recency_score_monotonic_decreasing(self):
        """Older interactions should give lower recency score."""
        scores = calculate_recency_score([1, 10, 100])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
    
    def test_interevent_score_range(self):
        """Inter-event score should be between 0 and 1."""
        scores = calculate_interevent_score([0.5, 5, 20, 50])
        for score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_composite_score_range(self):
        """Composite score should be between 0 and 1."""
        scores = calculate_relationship_score(
            n_interactions=[0, 10, 50],
            days_since_last_interaction=[365, 30, 1],
            median_interval_days=[90, 10, 1]
        )
        for score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_weights_sum_to_one(self):
        """Should raise assertion if weights don't sum to 1."""
        with self.assertRaises(AssertionError):
            calculate_relationship_score(
                n_interactions=[10],
                days_since_last_interaction=[10],
                median_interval_days=[10],
                wf=0.5, wr=0.4, wi=0.3  # Sums to 1.2
            )
    
    def test_interpret_score(self):
        """Test score interpretation."""
        self.assertEqual(interpret_score(0.1), "Weak or No Relationship Signal")
        self.assertEqual(interpret_score(0.3), "Moderate Relationship Signal")
        self.assertEqual(interpret_score(0.6), "Strong Relationship Signal")
        self.assertEqual(interpret_score(0.9), "Very Strong Relationship Signal")


if __name__ == "__main__":
    unittest.main()