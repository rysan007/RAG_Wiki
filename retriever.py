import json
import chromadb
from sentence_transformers import SentenceTransformer

class WikipediaRetriever:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.top_k = self.config['top_k']
        
        print("Loading embedding model.")
        self.model = SentenceTransformer(self.config['embedding_model_name'])
        
        print(f"Connecting to Chroma database at {self.config['vector_db_path']}...")
        self.client = chromadb.PersistentClient(path=self.config['vector_db_path'])
        self.collection = self.client.get_collection(self.config['collection_name'])

    def retrieve(self, query: str):
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        retrieved_contexts = []
        for i in range(len(documents)):
            retrieved_contexts.append({
                "text": documents[i],
                "title": metadatas[i]['title'],
                "url": metadatas[i]['url'],
                "distance": distances[i]
            })
            
        return retrieved_contexts