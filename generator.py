import os
import json
from openai import OpenAI
from retriever import WikipediaRetriever

class RAGGenerator:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "http://10.246.100.230/v1")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.config['generator_model_name']
        self.retriever = WikipediaRetriever(config_path)

    def construct_prompt(self, query: str, retrieved_contexts: list) -> str:
        context_str = ""
        for i, ctx in enumerate(retrieved_contexts):
            context_str += f"[{i+1}] title: {ctx['title']}, url: {ctx['url']}\n"
            context_str += f"    {ctx['text']}\n\n"

        return f"Context:\n{context_str}User query: {query}"

    def generate_answer(self, query: str) -> str:
        print(f"Retrieving context for: '{query}'")
        contexts = self.retriever.retrieve(query)
        prompt = self.construct_prompt(query, contexts)
        
        system_prompt = (
            "Answer using only the provided context. If the context does not "
            "contain the answer, say you do not know. Cite sources by title and url."
        )

        print(f"Generating answer from {self.model_name}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"