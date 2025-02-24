from collections import defaultdict
from typing import Any, List, Tuple

class ReciprocalRankFusionService:
    def __init__(self, k: int = 60):
        """
        Initialize the service with a constant K used in the RRF formula.
        
        Args:
            k (int): Constant used in the RRF formula (default is 60).
        """
        self.k = k

    def fuse(self, *ranked_lists: List[Any]) -> Tuple[List[Tuple[Any, float]], List[Any]]:
        """
        Fuse ranks from multiple IR systems using Reciprocal Rank Fusion (RRF).

        Args:
            *ranked_lists: Ranked results from different IR systems.
            
        Returns:
            A tuple containing:
                - A list of tuples (document, score) sorted by RRF score in descending order.
                - A list of sorted documents.
        """
        rrf_map = defaultdict(float)

        # Calculate RRF score for each item in each ranked list.
        for rank_list in ranked_lists:
            for rank, item in enumerate(rank_list, start=1):
                rrf_map[item] += 1 / (rank + self.k)

        # Sort items based on their RRF scores in descending order.
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        sorted_documents = [item for item, score in sorted_items]
        
        return sorted_items, sorted_documents
