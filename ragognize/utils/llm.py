import torch
import jinja2
from typing import List, Tuple
from transformers import QuantoConfig
from peft import PeftModel
from transformer_heads import load_lora_with_heads
from transformer_heads.util.helpers import get_model_params
from transformers.generation import GenerationConfig

class LLM:
    """
    Large Language Model (LLM) class for loading and initializing transformer models.

    Args:
        name (str): The name of the model.
        quantization (str, optional): One of "float8", "int8", "int4", "int2".
            Defaults to None.
        default_temperature (float, optional): The default temperature for sampling text.
            Defaults to 0.0.
        auto_load (bool, optional): Whether to automatically load the model.
            Defaults to False.
        add_generation_prompt (bool, optional): Whether to add certain strings to the prompt, e.g. appending "assistant:" or prepending today's date (LLM-specific).
            Defaults to True.
        ft_checkpoint_dir (str, optional): Directory of the peft model checkpoint to merge and load with the base model.
            Defaults to None.
    
    Attributes:
        name (str): The name of the model.
        quantization (str): The quantization method used for the model.
        default_temperature (float): The default temperature for sampling text.
        loaded (bool): Indicates whether the model is loaded.
        model (object): The loaded model object.
        tokenizer (object): The tokenizer object associated with the model.
        model_config (dict): Configuration parameters for the model.
        tokenizer_config (dict): Configuration parameters for the tokenizer.
        hf_token (str): Hugging Face API token for model access. This attribute is shared among all instances of the class.
        add_generation_prompt (bool): Indicates whether to add certain strings to the prompt, e.g. appending "assistant:" or prepending today's date (LLM-specific).
        ft_checkpoint_dir (str): Directory of the peft model checkpoint to merge and load with the base model.
    """

    hf_token: str = None

    def __init__(
        self,
        name: str,
        quantization: str = None,
        default_temperature: float = 0.0,
        auto_load: bool = False,
        add_generation_prompt: bool = True,
        ft_checkpoint_dir: str = None,
        load_with_heads: bool = False
    ) -> None:
        """
        Initialize LLM with the provided parameters.
        """
        self.name = name
        self.quantization = quantization
        self.default_temperature = default_temperature

        self.loaded = False
        self.model = None
        self.tokenizer = None
        self.add_generation_prompt = add_generation_prompt
        self.ft_checkpoint_dir = ft_checkpoint_dir
        self.load_with_heads = load_with_heads
        
        self.model_config = {
            "device_map": 'auto'
        }
        self.tokenizer_config = {
            "device_map": "auto"
        }

        if LLM.hf_token:
            self.model_config["token"] = LLM.hf_token
            self.tokenizer_config["token"] = LLM.hf_token

        if quantization:
            self.model_config["quantization_config"] = QuantoConfig(weights=quantization)

        if auto_load:
            self.load()
    
    def __enter__(self) -> None:
        """
        Try to load the model when used in a "with" statement.
        """
        self.load()
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """
        Try to close the model once a "with" statement is finished or has been abruptly terminated.
        """
        self.unload()

    def __str__(self) -> str:
        """
        Return a readable string representation for the instance of this class.
        """
        if self.quantization:
            return f"{self.name} ({self.quantization})"
        else:
            return self.name

    def load(self) -> None:
        """
        Load the model into memory. Once loaded, this function does not do anything, preventing the model from being loaded multiple times.
        """
        if not self.loaded:
            self.loaded = True
            self._load()

            if self.ft_checkpoint_dir:
                if self.load_with_heads:
                    model_class = get_model_params(self.name)["model_class"]
                    self.model = load_lora_with_heads(
                        model_class,
                        self.ft_checkpoint_dir,
                        device_map="auto", #{"": torch.cuda.current_device()},
                        new_emb_size=len(self.tokenizer)
                    )
                else:
                    self.model = PeftModel.from_pretrained(self.model, self.ft_checkpoint_dir).merge_and_unload()


    def unload(self) -> None:
        """
        Remove the model from memory. Once removed, this function does not do anything, preventing the model from being unloaded multiple times.
        """
        if self.loaded:
            self.loaded = False
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

    def tokenize(self, input_str: str, add_special_tokens=True) -> torch.Tensor:
        """
        Convert a string to its token ids.
        """
        return self.tokenizer([input_str], return_tensors="pt", add_special_tokens=add_special_tokens).to(self.model.device)

    def detokenize(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Convert token ids to the corresponding string.
        """
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)[0]

    def tokenize_chat(self, chat, add_generation_prompt: bool = None):
        """
        Tokenize a conversation chat.
        """
        if add_generation_prompt is None:
            add_generation_prompt = self.add_generation_prompt
        try:
            encodeds = self.tokenizer.apply_chat_template(chat, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        except jinja2.exceptions.TemplateError as err:
            # LLM does not support system message
            # Adding system message to start of first user message
            system_message = [msg for msg in chat if msg["role"] == "system"][0]["content"]
            new_chat = []
            for msg in chat:
                if msg["role"] != "system":
                    new_chat.append(msg)
            new_chat[0]["content"] = f'{system_message}\n{new_chat[0]["content"]}'

            # Try again
            encodeds = self.tokenizer.apply_chat_template(new_chat, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        return encodeds

    def generate(self, prompt: str, system: str = None, max_new_tokens=500, temperature=None, do_sample=False, internal_states_device=None, internal_states_for_new_tokens=True) -> Tuple[str, str, str, List[int]]:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The input prompt for text generation.
            system (str, optional): System message. Defaults to None.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
            temperature (float, optional): The temperature parameter for sampling. Defaults to None.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            internal_states_device (str, optional): If the string is set to a certain device (e.g., "cpu" or "cuda"), the intermediate activation values (MLP) and the contextualized embedding vectors are extracted from each token and layer and sent to the device.
            internal_states_for_new_tokens (bool, optional): True if the internal states should only be returned for newly generated tokens, otherwise False.

        Returns:
            Tuple[str, str, str, List[int]]: The full prompt (string with prompt, system message and special tokens), the full chat (string with prompt, system message, llm output and special tokens), the generated text of the LLM (skipping special tokens), and the start indices of each token.
                OR
            Tuple[str, str, str, List[int], Tuple[Tuple[torch.Tensor]], Tuple[Tuple[torch.Tensor]]]: When `get_internal_states` is true, contextualized embedding vectors and intermediate activation values are extracted and appended to the result.
        """
        # If temperature not set, then use default temperature
        temperature = temperature if temperature else self.default_temperature

        # Settings for generating the output
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "output_hidden_states": internal_states_device is not None,
            "return_dict_in_generate": internal_states_device is not None,
        }
        self.extend_generation_config(generation_config)

        # Convert text to tokens
        model_inputs = self.to_model_inputs(prompt=prompt, system=system)

        full_prompt = self.detokenize(model_inputs, False)

        # Generate output
        if internal_states_device is None:
            if self.load_with_heads:
                generated_ids = self.model.generate(torch.tensor(model_inputs.tolist()).to(self.model.device), generation_config=GenerationConfig(_pad_token_tensor=None, **generation_config)).sequences
            else:
                generated_ids = self.model.generate(torch.tensor(model_inputs.tolist()).to(self.model.device), **generation_config)
        else:
            # Older version of forked transformers (ability to extract internal states during generation)
            # output, cevs, iavs = self.model.generate(torch.tensor(model_inputs.tolist()).to(self.model.device), **generation_config)
            # generated_ids = output["sequences"]

            # # Move all tensors to the same device if needed
            # cevs = [[t.to(internal_states_device) if t.device != internal_states_device else t for t in layer] for layer in cevs]
            # # "Remove" the tuples and merge everything into one single Tensor
            # cevs = torch.stack([torch.stack(layer, dim=0) for layer in cevs], dim=0)

            # # Move all tensors to the same device if needed
            # iavs = [[t.to(internal_states_device) if t.device != internal_states_device else t for t in layer] for layer in iavs]
            # # "Remove" the tuples and merge everything into one single Tensor
            # iavs = torch.stack([torch.stack(layer, dim=0) for layer in iavs], dim=0)

            if self.load_with_heads:
                output = self.model.generate(torch.tensor(model_inputs.tolist()).to(self.model.device), generation_config=GenerationConfig(_pad_token_tensor=None, **generation_config))
            else:
                output = self.model.generate(torch.tensor(model_inputs.tolist()).to(self.model.device), **generation_config)
            generated_ids = output["sequences"]

        full_chat = self.detokenize(generated_ids, False)

        # Remove prompt tokens from output
        generated_ids = generated_ids[:, len(model_inputs[0]):]

        if internal_states_device is not None:
            cevs, iavs = self.get_internal_states(full_chat, internal_states_device)

            if internal_states_for_new_tokens:
                cevs = cevs[len(model_inputs[0]):]
                iavs = iavs[len(model_inputs[0]):]

        # Get indices of where each token starts in the string
        token_starts = []
        for i in range(generated_ids.shape[1]):
            token_starts.append(len(self.detokenize(generated_ids[:, :i], False)))

        if internal_states_device is None:
            # Return full_prompt, full_chat and generated ids as text without special tokens (e.g. eos or bos)
            return full_prompt, full_chat, self.detokenize(generated_ids, skip_special_tokens=True), token_starts
        else:
            return full_prompt, full_chat, self.detokenize(generated_ids, skip_special_tokens=True), token_starts, cevs, iavs
    
    def get_token_length(self, full_text: str) -> int:
        prompt_inputs = self.tokenize(full_text)["input_ids"]
        return prompt_inputs.size()[1]

    def get_internal_states(self, full_chat: str, internal_states_device: str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the internal states of the model.

        Args:
            full_chat (str): The full chat with the model (with special tokens!).
            internal_states_device (str): The device of the internal states that get returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: CEVs and IAVs with shape [number of tokens, number of layers, embedding size], but CEVs have one more additional "layer".
        """
        # Convert text to tokens
        model_inputs = self.tokenize(full_chat, add_special_tokens=False)["input_ids"]

        # Retrieve internal states
        output = self.model(torch.tensor(model_inputs.tolist()).to(self.model.device), output_hidden_states=True)

        iavs = ()
        for layer_i in range(len(self.model.model.layers)):
            iavs += (self.model.model.layers[layer_i].mlp.activation_values_from_inserted_code.squeeze(),)
        # Move all tensors to the same device if needed
        iavs = [[t.to(internal_states_device) if t.device != internal_states_device else t for t in layer] for layer in iavs]
        # "Remove" the tuples and merge everything into one single Tensor
        iavs = torch.stack([torch.stack(layer, dim=0) for layer in iavs], dim=0).transpose(0, 1)

        cevs = ()
        for layer_i in range(len(output.hidden_states)):
            cevs += (output.hidden_states[layer_i].squeeze(),)
        # Move all tensors to the same device if needed
        cevs = [[t.to(internal_states_device) if t.device != internal_states_device else t for t in layer] for layer in cevs]
        # "Remove" the tuples and merge everything into one single Tensor
        cevs = torch.stack([torch.stack(layer, dim=0) for layer in cevs], dim=0).transpose(0, 1)
        
        return cevs.detach(), iavs.detach()
    
    def to_model_inputs(self, prompt: str, system: str = None, llm_output: str = None) -> torch.Tensor:
        """
        Convert the prompt, system message, and LLM output into model inputs.

        Args:
            prompt (str): User input prompt.
            system (str, optional): System message. Defaults to None.
            llm_output (str, optional): LLM output. Defaults to None.

        Returns:
            torch.Tensor: Model inputs as tokenized tensors.
        """
        chat = []

        # Add system message to chat
        if system:
            chat.append({
                "role": "system",
                "content": system
            })

        # Add user's prompt to chat
        chat.append({
            "role": "user",
            "content": prompt
        })

        # Add output of LLM to chat
        if llm_output:
            chat.append({
                "role": "assistant",
                "content": llm_output
            })
            
        return self.tokenize_chat(chat)
    
    def _load(self) -> None:
        """
        Load the model into memory.
        """
        ...

    def extend_generation_config(self, generation_config: dict) -> None:
        """
        Extend the generation configuration with LLM specific settings.

        Args:
            generation_config (dict): Generation configuration dictionary.
        """
        ...

    def get_max_token_count_on_gpu(self, token_bounds=(0, 8192)) -> int:
        while True:
            token_count = (token_bounds[0] + token_bounds[1]) // 2
            if token_count == token_bounds[0]:
                break

            tokens = [22] * token_count
            try:
                self.model(torch.tensor([tokens]).to(self.model.device))
                token_bounds = (token_count, token_bounds[1])
            except RuntimeError as err:
                # GPU error
                token_bounds = (token_bounds[0], token_count)
        return token_count