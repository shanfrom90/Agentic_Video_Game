from typing import Tuple, List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
import os
from dotenv import load_dotenv


class GameRetrievalTool:    
    def __init__(self, db_path: str = "./games_db", collection_name: str = "games"):
         self.client = chromadb.PersistentClient(path=db_path)
        
         self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
         self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def retrieve_game(self, query: str, n_results: int = 3) -> Tuple[List[str], List[float]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            return documents, distances
        except Exception as e:
            print(f"Error retrieving from database: {e}")
            return [], []


class RetrievalEvaluationTool:    
    HIGH_CONFIDENCE_THRESHOLD = 0.65
    MEDIUM_CONFIDENCE_THRESHOLD = 0.75
    
    @staticmethod
    def evaluate_retrieval(
        distances: List[float],
        num_results: int = 3,
        use_average: bool = False
    ) -> Dict[str, Any]:
        if not distances:
            return {
                "confidence": "LOW",
                "score": 1.0,
                "rationale": "No results found in database"
            }
        
        if use_average:
            score = sum(distances[:num_results]) / min(len(distances), num_results)
        else:
            score = distances[0]
        
        if score < RetrievalEvaluationTool.HIGH_CONFIDENCE_THRESHOLD:
            confidence = "HIGH"
            rationale = f"Strong match found (distance: {score:.3f})"
        elif score < RetrievalEvaluationTool.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = "MEDIUM"
            rationale = f"Moderate match found (distance: {score:.3f})"
        else:
            confidence = "LOW"
            rationale = f"Weak or no match (distance: {score:.3f})"
        
        return {
            "confidence": confidence,
            "score": score,
            "rationale": rationale
        }


class WebSearchTool:    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        self.client = TavilyClient(api_key=self.api_key)
    
    def game_web_search(
        self,
        query: str,
        include_answer: bool = True,
        max_results: int = 5
    ) -> Dict[str, Any]:
        try:
            response = self.client.search(
                query=query,
                include_answer=include_answer,
                max_results=max_results
            )
            
            return {
                "success": True,
                "query": query,
                "results": response.get("results", []),
                "answer": response.get("answer", None),
                "response_time": response.get("response_time", None)
            }
        except Exception as e:
            print(f"Error during web search: {e}")
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": str(e)
            }