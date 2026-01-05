import numpy as np
from typing import Dict
import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent / "projects" / "minos-data-main"))
# sys.path.append(str(Path(__file__).parent.parent / "projects" / "minos-core-main"))

# from minos_core.dataset import RatingType  # Not used in this standalone version

ERROR_COST_MATRIX = np.array([
    [0, -1, -2],
    [1,  0, -1],
    [2,  1,  0]
])

RATING_INDEX_MAP = {
    1: 0,
    0: 1,
    -1: 2
}


class FaviScore:
    
    def __init__(self):
        self.idx_map = RATING_INDEX_MAP
        self.W = ERROR_COST_MATRIX
    

    # Create 3x3 confusion matrix
    def compute_confusion_matrix(self, human_ratings: np.ndarray, automated_ratings: np.ndarray) -> np.ndarray:
        C = np.zeros((3, 3), dtype=int)

        # Using zip to iterate over both arrays in parallel
        for h_rating, a_rating in zip(human_ratings, automated_ratings): 
            h_idx = self.idx_map[h_rating]
            a_idx = self.idx_map[a_rating]
            C[h_idx, a_idx] += 1
        return C
    
    # Compute FaviScore from confusion matrix
    def compute_faviscore(self, confusion_matrix: np.ndarray) -> float:
        total_errors = np.sum(confusion_matrix) - np.trace(confusion_matrix)
        if total_errors == 0:
            return 0.0
        weighted_sum = np.sum(self.W * confusion_matrix) # multiply both 3x3 matrices element-wise, then sum up the elements
        return weighted_sum / total_errors
    
    # Alternative FaviScore computation
    def compute_faviscore_alternative(self, human_ratings: np.ndarray, automated_ratings: np.ndarray) -> float:
        human_outcome = self._calculate_outcome_margin(human_ratings)
        automated_outcome = self._calculate_outcome_margin(automated_ratings)
        total_errors = np.sum(human_ratings != automated_ratings)
        if total_errors == 0:
            return 0.0
        return (automated_outcome - human_outcome) / total_errors
    
    # Calculate outcome margin
    def _calculate_outcome_margin(self, ratings: np.ndarray) -> float:
        # Count the number of +1's and -1's in the ratings array
        d_plus = np.sum(ratings == 1)
        d_minus = np.sum(ratings == -1)
        return d_plus - d_minus
    
    # Analyze bias and compute metrics
    def analyze_bias(self, human_ratings: np.ndarray, automated_ratings: np.ndarray) -> Dict:
        # Compute confusion matrix
        C = self.compute_confusion_matrix(human_ratings, automated_ratings)

        # Compute FaviScore using confusion matrix
        faviscore_matrix = self.compute_faviscore(C)

        # Compute FaviScore using outcome margin
        faviscore_alternative = self.compute_faviscore_alternative(human_ratings, automated_ratings)
        
        total_comparisons = len(human_ratings) # number of comparisons
        total_errors = np.sum(C) - np.trace(C) # trace is the sum of the diagonal elements

        # Calculate error rate
        if total_comparisons > 0:
            error_rate = total_errors / total_comparisons
        else:
            error_rate = 0

        # Calculate sample-level accuracy
        agreement = np.trace(C)
        if total_comparisons > 0:
            sample_level_accuracy = agreement / total_comparisons
        else:
            sample_level_accuracy = 0

        # Calculate system-level accuracy
        human_outcome = self._calculate_outcome_margin(human_ratings)
        automated_outcome = self._calculate_outcome_margin(automated_ratings)
        if np.sign(human_outcome) == np.sign(automated_outcome):
            system_level_accuracy = 1.0
        else:
            system_level_accuracy = 0.0
        
        return {
            'faviscore_matrix': faviscore_matrix,
            'faviscore_alternative': faviscore_alternative,
            'confusion_matrix': C,
            'total_comparisons': total_comparisons,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'sample_level_accuracy': sample_level_accuracy,
            'system_level_accuracy': system_level_accuracy,
            'human_outcome_margin': human_outcome,
            'automated_outcome_margin': automated_outcome,
            'error_cost_matrix': self.W
        }