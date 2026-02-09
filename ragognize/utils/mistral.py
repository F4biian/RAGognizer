from utils.llm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# For activation value extraction do the following:
# Install transformers via a special fork:
#   > pip install git+https://github.com/F4biian/transformers-v4.47.1-and-internal-states.git
#
#           OR
# 
# Go to the transformers package on your local disk (e.g. ~/.local/lib/python3.10/site-packages/transformers/models/mistral/modeling_mistral.py) and/or the associated modular script.
# In the MistralMLP class, alter the `forward` method this way:
'''
# Original version
# down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

# Version taken from MIND (https://github.com/oneal2000/MIND/issues/2):
a = self.act_fn(self.gate_proj(hidden_state))
self.activation_values_from_inserted_code = a.clone().detach()
down_proj = self.down_proj(a * self.up_proj(hidden_state))
'''

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
# Knowledge cutoff: max. May 22, 2024 (upload of .safetensors files)
class Mistral_7B_Instruct_v0_3(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False, ft_checkpoint_dir: str = None, load_with_heads: bool = False) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.3", quantization, default_temperature, auto_load, ft_checkpoint_dir=ft_checkpoint_dir, load_with_heads=load_with_heads)

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


# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# Knowledge cutoff: max. Dec 11, 2023 (upload of .safetensors files)
class Mistral_7B_Instruct_v0_1(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False, ft_checkpoint_dir: str = None, load_with_heads: bool = False) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", quantization, default_temperature, auto_load, ft_checkpoint_dir=ft_checkpoint_dir, load_with_heads=load_with_heads)

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