from .llama import Llama3_1_8B_Instruct, Llama2_7B_Chat_HF
from .mistral import Mistral_7B_Instruct_v0_3, Mistral_7B_Instruct_v0_1
from .llm import LLM

__all__ = [
    "Llama3_1_8B_Instruct",
    "Llama2_7B_Chat_HF",
    "Mistral_7B_Instruct_v0_3",
    "Mistral_7B_Instruct_v0_1",
    "set_hf_token",
    "StructuredOutputLLM"
]

def set_hf_token(token: str) -> None:
    LLM.hf_token = token