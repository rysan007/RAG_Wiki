import os
import glob
import json
from sentence_transformers import SentenceTransformer
import chromadb
from generator import RAGGenerator

def ingest_new_documents(config):
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return False

    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if len(txt_files) < 5:
        return False

    print(f"Found {len(txt_files)} new files. Initializing embedding model.")
    model = SentenceTransformer(config['embedding_model_name'])
    client = chromadb.PersistentClient(path=config['vector_db_path'])
    col = client.get_collection(config['collection_name'])

    texts, metadatas, ids = [], [], []

    for i, file_path in enumerate(txt_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            texts.append(content)
            filename = os.path.basename(file_path)
            metadatas.append({'title': filename, 'url': f"local://{filename}"})
            ids.append(f"custom_{i}_{filename}")

    print("Embedding new documents...")
    embs = model.encode(texts, show_progress_bar=True).tolist()

    print("Inserting into the existing Chroma database...")
    col.upsert(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
    print("Custom ingestion complete\n")
    return True

def main():
    print("--- Phase 3: Part 2 Pipeline ---")
    with open("config.json", "r") as f:
        config = json.load(f)
        
    if not ingest_new_documents(config):
        print("Failed adding new custom files.")
        return

    print("Initializing RAG Generator.")
    generator = RAGGenerator()

    # 10 Part 2 Queries (EverQuest topics on foods and locations)
    # Load queries from the external JSON file
    with open("queries.json", "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    queries = queries_data["part2"]

    results_table = []
    print("\nStarting Part 2 Inference.\n")

    for i, query in enumerate(queries):
        print(f"--- Processing Query {i+1}/10: '{query}' ---")
        contexts = generator.retriever.retrieve(query)
        answer = generator.generate_answer(query)
        
        top_k_sources = []
        for ctx in contexts:
            source_type = "New Item" if ctx['url'].startswith("local://") else "Starter Corpus"
            top_k_sources.append({
                "title": ctx['title'],
                "url": ctx['url'],
                "source_type": source_type,
                "distance": round(ctx['distance'], 4)
            })

        results_table.append({
            "query_id": f"P2_Q{i+1}",
            "query_text": query,
            "top_k_sources": top_k_sources,
            "generated_answer": answer,
            "grounded_correctly": "" 
        })
        print(f"Answer generated. Moving to next.\n")

    with open("part2_results.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=4)
        
    print(f"Part 2 complete. Results saved to part2_results.json")

if __name__ == "__main__":
    main()