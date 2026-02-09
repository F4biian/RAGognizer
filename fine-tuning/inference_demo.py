import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragognizer')))

from ragognizer.detectors.RAGognizer import RAGognizer

detector = RAGognizer(
    ragognizer_repo_name=None,
    checkpoint_dir="./fine-tuned_models/Qwen3-0.6B/checkpoint_3",
)

chat = [
    {
        "role": "user",
        "content": "Context: The wall is green.\nBased solely on the context: What color is the wall?"
    },
    {
        "role": "assistant",
        "content": "Thank you for reaching out! The color of the wall is gray."
    }
]

pred = detector.predict(chat=chat, token_level=True, verbose=True)
print(pred)

response, scores = detector.generate(chat=chat[:-1], return_hallucination_scores=True)
print(response)
print(scores)