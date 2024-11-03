import networkx as nx
import pandas as pd
from typing import Dict, Any
import redis
import json
from ..config import Config


class GraphFeatureCalculator:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True
        )

    def calculate_centrality_measures(
        self, npi: str, data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate or retrieve centrality measures for a provider."""
        # Try to get cached centrality measures
        cached_measures = self.redis_client.get(f"centrality:{npi}")
        if cached_measures:
            return json.loads(cached_measures)

        # If not cached, calculate measures
        G = self._build_graph(data)
        measures = {
            "HCPCS_Degree_Centrality": nx.degree_centrality(G)[npi],
            "HCPCS_Closeness_Centrality": nx.closeness_centrality(G, distance="weight")[
                npi
            ],
            "HCPCS_PageRank": nx.pagerank(G, weight="weight")[npi],
            "Provider Type_Closeness_Centrality": nx.closeness_centrality(
                G, distance="weight"
            )[npi],
            "Provider Type_PageRank": nx.pagerank(G, weight="weight")[npi],
        }

        # Cache the results
        self.redis_client.setex(
            f"centrality:{npi}",
            3600,  # Cache for 1 hour
            json.dumps(measures),
        )

        return measures

    def _build_graph(self, data: Dict[str, Any]) -> nx.Graph:
        """Build provider network graph from data."""
        G = nx.Graph()
        # Add edges and weights based on the data
        # This is a simplified version - in production, you'd use historical data
        G.add_edge(data["Rndrng_NPI"], data["HCPCS_Cd"], weight=data["Avg_Sbmtd_Chrg"])
        return G
