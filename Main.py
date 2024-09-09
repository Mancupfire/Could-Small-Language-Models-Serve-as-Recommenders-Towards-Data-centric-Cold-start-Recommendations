import numpy as np
from typing import List, Tuple, Dict
from scipy.special import softmax

class PromptRec:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.language_model = self.initialize_language_model()

    def initialize_language_model(self) -> Dict[str, np.ndarray]:
        return {
            'embeddings': np.random.randn(self.vocab_size, self.embedding_dim),
            'output_layer': np.random.randn(self.embedding_dim, self.vocab_size)
        }

    def compute_preference_score(self, user: int, item: int) -> float:
        # Equation (1) in the paper
        v_pos = self.language_model['embeddings'][item]  # positive
        v_neg = np.mean(self.language_model['embeddings'], axis=0)  # negative
        
        numerator = np.exp(np.dot(self.language_model['embeddings'][user], v_pos))
        denominator = numerator + np.exp(np.dot(self.language_model['embeddings'][user], v_neg))
        
        return numerator / denominator

# Equation (2) in the paper
    def estimate_probability(self, context: List[int], target: int) -> float:
        context_embed = np.mean([self.language_model['embeddings'][w] for w in context], axis=0)
        logits = np.dot(context_embed, self.language_model['output_layer'])
        probs = softmax(logits)
        return probs[target]

    def prompt_refinement(self, user: int, item: int, prompt_template: str) -> str:
        filled_prompt = prompt_template.replace('[USER]', f"User_{user}")
        filled_prompt = filled_prompt.replace('[ITEM]', f"Item_{item}")
        return filled_prompt

    def extract_refined_corpus(self, general_corpus: List[str], cold_start_corpus: List[str]) -> List[str]:
        # Equation (8) in the paper
        c_star = []
        for doc in general_corpus:
            mi_score = self.mutual_information(doc, cold_start_corpus)
            if mi_score > 0:  
                c_star.append(doc)
        return c_star

    def mutual_information(self, doc: str, corpus: List[str]) -> float:
        doc_words = set(doc.split())
        corpus_words = set(word for text in corpus for word in text.split())
        
        intersection = len(doc_words.intersection(corpus_words))
        return intersection / (len(doc_words) + len(corpus_words) - intersection)

    def transferable_prompt_pretraining(self, source_datasets: List[List[Tuple[int, int]]], target_dataset: List[Tuple[int, int]]):
        # Equation (11) in the paper
        all_data = sum(source_datasets, []) + target_dataset
        user_embeddings = np.random.randn(len(set([u for u, _ in all_data])), self.embedding_dim)
        item_embeddings = np.random.randn(len(set([i for _, i in all_data])), self.embedding_dim)

        for epoch in range(10):  
            for user, item in all_data:
                user_embed = user_embeddings[user]
                item_embed = item_embeddings[item]
                score = self.compute_preference_score(user, item)
                
              
                user_embeddings[user] += 0.01 * (1 - score) * item_embed
                item_embeddings[item] += 0.01 * (1 - score) * user_embed

        
        self.language_model['embeddings'] = np.vstack([user_embeddings, item_embeddings])

    def cold_start_recommendation(self, user: int, items: List[int]) -> List[Tuple[int, float]]:
        # Equation (3) in the paper
        scores = []
        for item in items:
            r_ui = self.compute_preference_score(user, item)
            scores.append((item, r_ui))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def prompt_decomposition(self, prompt: str) -> Tuple[str, str]:
        words = prompt.split()
        mid = len(words) // 2
        return ' '.join(words[:mid]), ' '.join(words[mid:])

# Demo usage
vocab_size = 10000
embedding_dim = 128
recommender = PromptRec(vocab_size, embedding_dim)

# Data Simulation
users = list(range(100))
items = list(range(1000))
interactions = [(np.random.choice(users), np.random.choice(items)) for _ in range(10000)]

# Pretrain the model
source_datasets = [interactions[:5000], interactions[5000:8000]]
target_dataset = interactions[8000:]
recommender.transferable_prompt_pretraining(source_datasets, target_dataset)

# MCold-start recommendation
new_user = 101
candidate_items = np.random.choice(items, 10, replace=False)
recommendations = recommender.cold_start_recommendation(new_user, candidate_items)

print("Top 3 recommendations for the new user:")
for item, score in recommendations[:3]:
    print(f"Item {item}: Score {score:.4f}")

# Demonstrate prompt refinement and decomposition
prompt_template = "Recommend [ITEM] to [USER] based on their preferences."
refined_prompt = recommender.prompt_refinement(new_user, recommendations[0][0], prompt_template)
task_prompt, domain_prompt = recommender.prompt_decomposition(refined_prompt)

print("\nRefined Prompt:", refined_prompt)
print("Task Prompt:", task_prompt)
print("Domain Prompt:", domain_prompt)
