
from typing import Dict

class CostTracker:
    def __init__(self):
        # Initialize cost counters
        self.total_read_units = 0
        self.total_write_units = 0
        self.total_llm_usage = None
        self.total_embedding_tokens = 0
        self.rerank_units = 0
        self.other_costs = {}  

    def add_read_units(self, units: int):
        self.total_read_units += units

    def add_write_units(self, units: int):
        self.total_write_units += units

    def add_llm_tokens(self, llm_usage: dict):
        self.total_llm_usage = llm_usage

    def add_embedding_tokens(self, tokens: int):
        self.total_embedding_tokens += tokens

    def add_custom_cost(self, key: str, cost: float):
        self.other_costs[key] = self.other_costs.get(key, 0) + cost
    def add_rerank_units(self, rerank_units : int):
        self.rerank_units += rerank_units
        
    def to_dict(self) -> Dict:
        return {
            "total_read_units": self.total_read_units,
            "total_write_units": self.total_write_units,
            "total_llm_usage": self.total_llm_usage,
            "total_embedding_tokens": self.total_embedding_tokens,
            "other_costs": self.other_costs,
            "rerank_units" : self.rerank_units
        }
        
cost_tracker = CostTracker()