import json
from generator import RAGGenerator

def main():
    print("Initializing RAG Generator")
    generator = RAGGenerator()

    queries = [
        "What is a Artificial intelligence?",  # start off with easy questions
        "Who is the father of video games?",
        "What is the capitol of Texas?",
        "What is the largest country?",
        "What is the largest ocean on Earth?",
        "Who wrote the James Bond series?", # More specific question
        "What is the Mario Brothers series about?", # popular game, might know info
        "Who is the largest exporter of coffee?", # Deeper dive - might not be explain in short passage
        "How large is our galaxy?", # Has to assume our galaxy
        "What is the most popular class in EverQuest?"  # Very specific, likely not in data
    ]

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