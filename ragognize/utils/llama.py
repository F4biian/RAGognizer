from utils.llm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# For activation value extraction do the following:
# Install transformers via a special fork:
#   > pip install git+https://github.com/F4biian/transformers-v4.47.1-and-internal-states.git
#
#           OR
# 
# Go to the transformers package on your local disk (e.g. ~/.local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py) and/or the associated modular script.
# In the LlamaMLP class, alter the `forward` method this way:
'''
# Original version
# down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Version taken from MIND (https://github.com/oneal2000/MIND/issues/2):
a = self.act_fn(self.gate_proj(x))
self.activation_values_from_inserted_code = a.clone().detach()
down_proj = self.down_proj(a * self.up_proj(x))
'''

# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Knowledge cutoff: December 2023
class Llama3_1_8B_Instruct(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False, ft_checkpoint_dir: str = None, load_with_heads: bool = False) -> None:
        super().__init__("meta-llama/Llama-3.1-8B-Instruct", quantization, default_temperature, auto_load, ft_checkpoint_dir=ft_checkpoint_dir, load_with_heads=load_with_heads)

    def _load(self) -> None:
        if not self.load_with_heads:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_config)

        if self.ft_checkpoint_dir:
            # Add pad_token to tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
                self.tokenizer.padding_side = "right"
                if not self.load_with_heads:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.model.resize_token_embeddings(len(self.tokenizer))

    def extend_generation_config(self, generation_config: dict) -> None:
        if self.ft_checkpoint_dir:
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        else:
            generation_config["pad_token_id"] = self.tokenizer.eos_token_id


# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# Knowledge cutoff: July 2023 (because of more recent tuning data)
class Llama2_7B_Chat_HF(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False, ft_checkpoint_dir: str = None, load_with_heads: bool = False) -> None:
        super().__init__("meta-llama/Llama-2-7b-chat-hf", quantization, default_temperature, auto_load, ft_checkpoint_dir=ft_checkpoint_dir, load_with_heads=load_with_heads)

    def _load(self) -> None:
        if not self.load_with_heads:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_config)

        if self.ft_checkpoint_dir:
            # Add pad_token to tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.tokenizer.padding_side = "right"
                if not self.load_with_heads:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.model.resize_token_embeddings(len(self.tokenizer))

    def extend_generation_config(self, generation_config: dict) -> None:
        if self.ft_checkpoint_dir:
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        else:
            generation_config["pad_token_id"] = self.tokenizer.eos_token_id