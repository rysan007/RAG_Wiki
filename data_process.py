import os
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

def main():
    # Load Config
    with open("config.json", "r") as f:
        config = json.load(f)

    print("Initializing Model and Database.")
    model = SentenceTransformer(config['embedding_model_name']) 

    # Setup Chroma client
    client = chromadb.PersistentClient(path=config['vector_db_path'])
    col = client.get_or_create_collection(config['collection_name'])

    print("Loading Raw Wikipedia Dataset")
    ds = load_dataset('wikipedia', '20220301.simple', split='train', streaming=True, trust_remote_code=True)

    print("Extracting paragraphs from articles")
    texts = []
    metadatas = []
    ids = []
    target_size = 10000
    passage_count = 0

    for row in ds:
        paragraphs = row['text'].split('\n\n')
        for p in paragraphs:
            clean_p = p.strip()
            if 150 < len(clean_p) < 1500:
                texts.append(clean_p)
                metadatas.append({'title': row['title'], 'url': row['url']})
                ids.append(f"wiki_{passage_count}")
                passage_count += 1 
                break 
        if passage_count >= target_size:
            break

    print(f"Successfully extracted {len(texts)} short passages.")
    print("Embedding passages")
    embs = model.encode(texts, show_progress_bar=True).tolist() 

    print("Inserting data into Chroma in batches")
    # Chunk the inserts 
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        print(f"Inserting batch {i} to {i + batch_size}...")
        col.add(
            ids=ids[i:i+batch_size], 
            embeddings=embs[i:i+batch_size], 
            documents=texts[i:i+batch_size], 
            metadatas=metadatas[i:i+batch_size]
        )
    
    print(f"Complete! Database saved to {os.path.abspath(config['vector_db_path'])}")

if __name__ == "__main__":
    main()