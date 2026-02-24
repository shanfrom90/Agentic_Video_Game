from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.utils import embedding_functions
import json
import os


class VectorStoreManager:
    def __init__(self, db_path: str = "./games_db", collection_name: str = "games"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        print(f"✓ VectorStoreManager initialized")
        print(f"  Database path: {db_path}")
        print(f"  Collection: {collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": "all-MiniLM-L6-v2"
        }
    
    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✓ Collection '{self.collection_name}' cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")


class DataProcessor:
    @staticmethod
    def load_game_data(folder_path: str) -> List[Dict[str, Any]]:
        games = []
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        
        for file in json_files:
            try:
                with open(os.path.join(folder_path, file), 'r') as f:
                    data = json.load(f)
                    games.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"✓ Loaded {len(games)} games from {folder_path}")
        return games
    
    @staticmethod
    def create_documents(games: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        documents = []
        metadatas = []
        ids = []
        
        for i, game in enumerate(games):
            platforms = game.get('platforms') or []
            if not isinstance(platforms, list):
                platforms = [str(platforms)]
            
            text = f"""Title: {game.get('title')}
Genre: {game.get('genre')}
Release Date: {game.get('release_date')}
Platforms: {', '.join(platforms)}
Publisher: {game.get('publisher')}
Description: {game.get('description')}"""
            
            documents.append(text)
            metadatas.append({"title": game.get("title")})
            ids.append(str(i))
        
        print(f"✓ Created {len(documents)} documents from games")
        return documents, metadatas, ids


class SemanticSearch:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store = vector_store_manager
    
    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        try:
            results = self.vector_store.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            return {
                "query": query,
                "documents": documents,
                "distances": distances,
                "metadatas": metadatas,
                "num_results": len(documents)
            }
        except Exception as e:
            print(f"Error during search: {e}")
            return {
                "query": query,
                "documents": [],
                "distances": [],
                "metadatas": [],
                "num_results": 0,
                "error": str(e)
            }
    
    def display_results(self, search_result: Dict[str, Any]) -> None:
        print("\n" + "="*80)
        print(f"SEARCH QUERY: {search_result['query']}")
        print("="*80)
        
        if search_result['num_results'] == 0:
            print("No results found.")
            return
        
        for i, (doc, distance, metadata) in enumerate(
            zip(search_result['documents'], search_result['distances'], search_result['metadatas'])
        ):
            print(f"\nResult {i+1} - {metadata.get('title', 'Unknown')} (Distance: {distance:.3f})")
            print("-" * 80)
            print(doc)


class RAGPipeline:
    def __init__(self, db_path: str = "./games_db"):
        self.vector_store = VectorStoreManager(db_path=db_path)
        self.data_processor = DataProcessor()
        self.semantic_search = SemanticSearch(self.vector_store)
    
    def setup_pipeline(self, data_folder: str) -> Dict[str, Any]:
        print("\n" + "="*80)
        print("RAG PIPELINE SETUP")
        print("="*80)
        
        games = self.data_processor.load_game_data(data_folder)
        documents, metadatas, ids = self.data_processor.create_documents(games)
        
        print(f"\n✓ Adding {len(documents)} documents to vector store...")
        self.vector_store.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        stats = self.vector_store.get_collection_stats()
        print(f"✓ Vector store populated with {stats['document_count']} documents")
        print(f"  Embedding model: {stats['embedding_model']}")
        
        return {
            "total_games": len(games),
            "total_documents": len(documents),
            "vector_store_stats": stats
        }
    
    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        return self.semantic_search.search(query, n_results)
    
    def display_search_results(self, search_result: Dict[str, Any]) -> None:
        self.semantic_search.display_results(search_result)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_collection_stats()