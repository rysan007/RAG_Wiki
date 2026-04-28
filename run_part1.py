import json
from generator import RAGGenerator

def main():
    print("Initializing RAG Generator")
    generator = RAGGenerator()

    # Load queries from the external JSON file
    with open("queries.json", "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    queries = queries_data["part1"]

    results_table = []
    print("\nStarting Part 1 Inference...\n")

    for i, query in enumerate(queries):
        print(f"--- Processing Query {i+1}/10: '{query}' ---")
        contexts = generator.retriever.retrieve(query)
        answer = generator.generate_answer(query)
        
        top_k_sources = []
        for ctx in contexts:
            top_k_sources.append({
                "title": ctx['title'],
                "url": ctx['url'],
                "distance": round(ctx['distance'], 4)
            })

        results_table.append({
            "query_id": f"Q{i+1}",
            "query_text": query,
            "top_k_sources": top_k_sources,
            "generated_answer": answer,
            "grounded_correctly": ""  
        })
        print(f"Answer generated. Moving to next.\n")

    with open("part1_results.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=4)
        
    print(f"Part 1 complete. Results saved to part1_results.json")

if __name__ == "__main__":
    main()