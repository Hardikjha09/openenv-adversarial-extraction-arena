"""
Elo Rating implementation for calculating relative Agent performance.
"""

import math
from typing import Dict, Tuple

class EloRater:
    def __init__(self, k_factor: float = 32.0, base_rating: float = 1200.0):
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {
            "extractor": base_rating,
            "adversary": base_rating
        }

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for A when playing B."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update(self, extractor_score: float, adversary_score: float) -> Tuple[float, float]:
        """
        Updates Elo ratings.
        Score normalized to [0, 1]. E.g. If Extractor parses perfectly despite Adversary edit,
        Extractor gets score=1, Adversary gets score=0.
        """
        # We assume scores are already normalized between 0 and 1
        extractor_rating = self.ratings["extractor"]
        adversary_rating = self.ratings["adversary"]
        
        expected_extractor = self.expected_score(extractor_rating, adversary_rating)
        expected_adversary = self.expected_score(adversary_rating, extractor_rating)
        
        # If both get high reward, it's a draw of sorts, but typically it's zero-sum
        # We will treat the "match score" for Extractor as its reward, 
        # and Adversary score as 1 - Extractor reward.
        
        match_score_ext = extractor_score
        match_score_adv = 1.0 - extractor_score # Simplified zero-sum
        
        new_ext = extractor_rating + self.k_factor * (match_score_ext - expected_extractor)
        new_adv = adversary_rating + self.k_factor * (match_score_adv - expected_adversary)
        
        self.ratings["extractor"] = new_ext
        self.ratings["adversary"] = new_adv
        
        return new_ext, new_adv
