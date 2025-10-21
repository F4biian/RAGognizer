from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

import os
import gc
import jinja2
from typing import Literal
import torch
from ragognizer.detectors.detector import HallucinationDetector
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
from peft import PeftModel
import json
from transformer_heads import load_lora_with_heads
from transformer_heads.util.helpers import get_model_params
from transformer_heads.constants import model_type_map as TRANSFORMER_HEADS_ARCH_MAP


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)

        self.config = {
            "input_size": input_size,
            "hidden_dims": hidden_dims,
        }

    def forward(self, x):
        return self.network(x)


class RAGognizer(HallucinationDetector):
    def __init__(
        self,
        ragognizer_repo_name: Literal[
            "F4biian/RAGognizer-Qwen3-4B-Instruct-2507",
            "F4biian/RAGognizer-Llama-2-7b-chat-hf",
            "F4biian/RAGognizer-Llama-3.1-8B-Instruct",
            "F4biian/RAGognizer-Mistral-7B-Instruct-v0.1",
            "F4biian/RAGognizer-Mistral-7B-Instruct-v0.3",
        ] = "F4biian/RAGognizer-Qwen3-4B-Instruct-2507",
        checkpoint_dir: str = None,
        device: str = "cuda",
        use_transformer_heads_library: bool=True
    ) -> None:
        """
        Initialize the RAGognizer detector.

        Args:
            ragognizer_repo_name (str, optional): The name of the RAGognizer repository. Defaults to None.
            checkpoint_dir (str, optional): The directory of the checkpoint. Defaults to None.
            device (str, optional): The device to run the model on. Defaults to "cuda".
            use_transformer_heads_library (bool, optional): Whether to use the transformer_heads library. Defaults to True.

        Raises:
            ValueError: If neither or both of ragognizer_repo_name and checkpoint_dir are provided
        """
        if ragognizer_repo_name is None and checkpoint_dir is None:
            raise ValueError("Please provide either ragognizer_repo_name or checkpoint_dir.")
        if ragognizer_repo_name is not None and checkpoint_dir is not None:
            raise ValueError("Please provide only one of ragognizer_repo_name or checkpoint_dir, not both.")

        self.ragognizer_repo_name = ragognizer_repo_name
        self.use_transformer_heads = use_transformer_heads_library
        if checkpoint_dir is None:
            repo_dir = snapshot_download(ragognizer_repo_name, repo_type="model")
            checkpoint_dir = os.path.join(repo_dir, "ft_llm")

        with open(os.path.join(checkpoint_dir, "adapter_config.json"), "r") as file:
            adapter_config = json.load(file)

        llm_name = adapter_config["base_model_name_or_path"].split("/")[-1]
        name = f"RAGognizer ({llm_name})"
        if not self.use_transformer_heads:
            name += " (sep. MLP)"
        super().__init__(name=name, aggregate_response_level_from_token_level=True)

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_config["base_model_name_or_path"], device_map=device)

        if self.use_transformer_heads:
            # Get base model class
            cfg = AutoConfig.from_pretrained(adapter_config["base_model_name_or_path"]).to_dict()
            arch = cfg["architectures"][0]

            # e.g. turn "Qwen3ForCausalLM" into "Qwen3Model" (might not work for every single model, but majority)
            arch = arch.replace("ForCausalLM", "Model")
            if cfg["model_type"] not in TRANSFORMER_HEADS_ARCH_MAP:
                _class = getattr(
                    __import__("transformers", fromlist=[arch]),
                    arch,
                )
                # Ensure that model type is mapped to architecture
                TRANSFORMER_HEADS_ARCH_MAP[cfg["model_type"]] = ("model", _class)

            # Check for blocked terms
            blocklist = ["ForQuestionAnswering", "ForCausalLM", "PreTrained", "ForSequenceClassification", "ForTokenClassification"]
            for b in blocklist:
                if b in TRANSFORMER_HEADS_ARCH_MAP[cfg["model_type"]]:
                    raise Exception(f"Model architecture {arch} not supported in transformer_heads library. The model_type_map does not contain a valid base model class for it, as it contains a blocked term: {b}. For instance, for Qwen3 models, it should be 'Qwen3Model' and not 'Qwen3ForCausalLM', 'Qwen3ForCausalLM', or 'Qwen3PreTrainedModel' or similar.")

            model_params = get_model_params(adapter_config["base_model_name_or_path"])
            model_class = model_params["model_class"]

            try:
                other_data_path  = os.path.join(repo_dir, "other_data.json")
                with open(other_data_path, "r") as f:
                    other_data = json.load(f)
            except:
                try:
                    other_data_path  = os.path.join(repo_dir, "mlp_other_data.json")
                    with open(other_data_path, "r") as f:
                        other_data = json.load(f)
                except:
                    other_data = {}

            # Account for a potentially added PAD token (offset of 1)
            embeddings_offset = other_data.get("new_embeddings_added", 0)
            self.llm = load_lora_with_heads(
                model_class,
                checkpoint_dir,
                device_map=device,
                new_emb_size=(len(self.tokenizer)+embeddings_offset) if embeddings_offset > 0 else None,
            )
            self.binarization_threshold = other_data.get("binarization_threshold", None)
        else:
            weights_path = os.path.join(repo_dir, "mlp_state.pt")

            if not os.path.exists(weights_path):
                raise ValueError(
                    "The provided checkpoint_dir does not contain the required MLP weights (mlp_state.pt), which are needed when not using the transformer_heads library."
                )

            config_path  = os.path.join(repo_dir, "mlp_config.json")
            other_data_path  = os.path.join(repo_dir, "mlp_other_data.json")
            with open(config_path, "r") as f:
                mlp_cfg = json.load(f)
            with open(other_data_path, "r") as f:
                other_data = json.load(f)

            self.llm = AutoModelForCausalLM.from_pretrained(other_data["original_repo_id"], device_map=device)
            self.llm = PeftModel.from_pretrained(self.llm, checkpoint_dir).merge_and_unload()

            self.mlp = MLP(**mlp_cfg)
            self.mlp.load_state_dict(torch.load(weights_path))
            self.mlp.to(device).eval()
            
            self.layer_perc = other_data["layer_perc"]
            self.binarization_threshold = other_data["binarization_threshold"]

        self.llm.requires_grad_(False)
        self.llm.eval()
    
    def tokenize_chat(self, chat, add_generation_prompt: bool = False):
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

    def detokenize(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)[0]
    
    def tokenize(self, input_str: str, add_special_tokens=True) -> torch.Tensor:
        return self.tokenizer([input_str], return_tensors="pt", add_special_tokens=add_special_tokens).to(self.llm.device)

    def get_internal_states(self, full_chat: str, internal_states_device: str="cpu") -> torch.Tensor:
        """
        Retrieve the internal states of the model.

        Args:
            full_chat (str): The full chat with the model (with special tokens!).
            internal_states_device (str): The device of the internal states that get returned.

        Returns:
            torch.Tensor: CEVs with shape [number of tokens, number of layers, embedding size], but CEVs have one more additional "layer".
        """
        # Convert text to tokens
        model_inputs = self.tokenize(full_chat, add_special_tokens=False)["input_ids"]

        # Retrieve internal states
        output = self.llm(torch.tensor(model_inputs.tolist()).to(self.llm.device), output_hidden_states=True)

        cevs = ()
        for layer_i in range(len(output.hidden_states)):
            cevs += (output.hidden_states[layer_i].squeeze(),)
        # Move all tensors to the same device if needed
        cevs = [[t.to(internal_states_device) if t.device != internal_states_device else t for t in layer] for layer in cevs]
        # "Remove" the tuples and merge everything into one single Tensor
        cevs = torch.stack([torch.stack(layer, dim=0) for layer in cevs], dim=0).transpose(0, 1)
        
        del output

        return cevs.detach()

    def _pack_probs(
        self,
        probs,
        input_ids,
        original_chat,
        return_assistant_only,
        only_readable_tokens,
    ):
        token_spans = []
        for ti in range(probs.shape[0]):
            tok = self.detokenize([input_ids[0, ti]])
            tokens_str = self.detokenize([input_ids[0, :ti+1]])
            if len(tok) > 0:
                prob = probs[ti].item()
                if self.binarization_threshold is not None:
                    pred = 1 if prob >= self.binarization_threshold else 0
                else:
                    pred = None
                token_spans.append({
                    "start": len(tokens_str) - len(tok),
                    "text": tok,
                    "end": len(tokens_str),
                    "prob": prob,
                    "pred": pred,
                })

        if return_assistant_only:
            try:
                assistant_starts_at = self.detokenize(input_ids).rindex(original_chat[-1]["content"])
            except:
                # For some models whitespace gets removed, so we do stripping to find start
                assistant_starts_at = self.detokenize(input_ids).rindex(original_chat[-1]["content"].strip())

            new_spans = []
            for span in token_spans:
                if span["start"] >= assistant_starts_at:
                    span["start"] -= assistant_starts_at
                    span["end"]   -= assistant_starts_at
                    new_spans.append(span)
            token_spans = new_spans
        
        if only_readable_tokens and return_assistant_only:
            response = None
            for msg in original_chat:
                if msg["role"] == "assistant":
                    response = msg["content"]
                    break
            else:
                print(f"WARNING: Could not find assistant's message in the following chat:\n{original_chat}")
                return token_spans

            # Remove tokens that are not part of the real response (e.g. "[SEP]")
            return self._ensure_tokens_in_response(response=response, token_prediction=token_spans)
        else:
            return token_spans

    def _predict(
        self,
        chat: list[dict[str, str]] = None,
        # ---*** kwargs below ***---
        return_assistant_only: bool = True,
        only_readable_tokens: bool = True,
        verbose: bool = False
    ) -> list[dict[str, float | str | int]]:
        original_chat = chat

        model_inputs = self.tokenize_chat(chat, add_generation_prompt=False)
        chat = self.detokenize(ids=model_inputs, skip_special_tokens=False)

        model_inputs = self.tokenize(chat, add_special_tokens=False)
        input_ids = model_inputs["input_ids"]

        if self.use_transformer_heads:
            outputs = self.llm(
                **model_inputs
            )
            preds_per_token = None
            for key in outputs.preds_by_head:
                if "hallu" in key:
                    preds_per_token = outputs.preds_by_head[key].flatten()
                    break
            probs = torch.sigmoid(preds_per_token).cpu().detach().numpy()
        else:
            cevs = self.get_internal_states(chat, next(self.mlp.parameters()).device)

            layer_i = int((cevs.shape[1] - 1) * self.layer_perc) + 1
            states = cevs[:, layer_i, :]

            logits = self.mlp(states)
            probs = torch.sigmoid(logits).detach().cpu().flatten()

        if verbose:
            text_row = ""
            prob_row = ""
            max_row_length = 100
            for ti in range(probs.shape[0]):
                tok = self.tokenizer.convert_ids_to_tokens([input_ids[0, ti]])[0]
                p = float(probs[ti])
                p_text = f"{int(p * 100)}"

                text_len = max(len(p_text), len(tok)) + 1

                tok = " " * (text_len - len(tok)) + tok
                p_text = " " * (text_len - len(p_text)) + p_text

                if len(text_row) + len(tok) > max_row_length:
                    print(text_row)
                    print(prob_row)
                    print()
                    prob_row = ""
                    text_row = ""

                text_row += tok
                prob_row += p_text

            print(text_row)
            print(prob_row)

        # Pack probs and tokens into a readable and nice dict/list format
        to_return = self._pack_probs(
            probs=probs,
            input_ids=input_ids,
            original_chat=original_chat,
            return_assistant_only=return_assistant_only,
            only_readable_tokens=only_readable_tokens
        )

        del model_inputs, input_ids, probs
        if 'cevs' in locals():
            del cevs
        if 'states' in locals():
            del states
        if 'logits' in locals():
            del logits

        torch.cuda.empty_cache()
        gc.collect()
        
        return to_return
    
    def predict(
        self,
        documents: list[str] = None,
        document: str = None,
        user: str = None,
        response: str = None,
        chat: list[dict[str, str]] = None,
        token_level: bool=True,
        # ---*** kwargs below ***---
        return_assistant_only: bool = True,
        verbose: bool = False
    ) -> list[dict[str, float | str | int]] | dict[str, float | int]:
        """
        Predict token-level or response-level hallucinations in the given input.

        You can either provide:
        - documents + user + response (will be converted to chat internally)
        - document + user + response (will be converted to chat internally)
        - chat (full chat history; only the last message will be analyzed if return_assistant_only is True)
        
        Args:
            documents (list[str], optional): List of document strings providing context. Defaults to None.
            document (str, optional): Single document string providing context. Defaults to None.
            user (str, optional): User message string. Defaults to None.
            response (str, optional): Assistant response string. Defaults to None.
            chat (list[dict[str, str]], optional): Full chat history as a list of messages. Each message is a dict with 'role' and 'content'. Defaults to None.
            token_level (bool, optional): Whether to return token-level predictions. If False, returns aggregated response-level prediction. Defaults to True.
            return_assistant_only (bool, optional): Whether to return predictions only for the assistant's response. Defaults to True.
            verbose (bool, optional): Whether to print detailed prediction information. Defaults to False.
        Returns:
            list[dict[str, float | str | int]] | dict[str, float | int]: 
                Token-level predictions if token_level is True, else response-level prediction.
        """
        if documents is None and document is not None:
            documents = [document]

        # If chat is not given, but other parts, then build an artificial chat
        if (
            chat is None and \
            documents is not None and \
            response is not None
        ):
            context = "\n\n".join(documents)
            user_message = f"""<context>{context}</context>"""
            if user is not None:
                user_message += f"\n\n<user>{user}</user>"

            chat = [
                # Ignoring system message
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]

        # If required input is missing, cannot predict (-> return None)
        if chat is None:
            return None
        
        pred = self._predict(
            chat=chat,
            return_assistant_only=return_assistant_only,
            verbose=verbose
        )

        if not pred:
            return None

        if token_level:
            return pred
        else:
            return self._aggregate_to_response_level(token_level_pred=pred)
        
    def generate(
        self,
        chat: list[dict[str, str]],
        max_new_tokens : int= 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        return_hallucination_scores: bool = True,
    ):
        if self.ragognizer_repo_name in [
            "F4biian/RAGognizer-Llama-3.1-8B-Instruct",
            "F4biian/RAGognizer-Mistral-7B-Instruct-v0.1",
            "F4biian/RAGognizer-Mistral-7B-Instruct-v0.3",
        ]:
            raise Exception(
                f"We don't recommend using `{self.ragognizer_repo_name}` for generations, since it was observed that it lost certain language capabilities."
            )

        gen_config = GenerationConfig(
            _pad_token_tensor=None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id
        )

        if self.use_transformer_heads:
            self.llm.generation_config = gen_config

        input_ids = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
        output = self.llm.generate(torch.tensor(input_ids.tolist()).to("cuda"), generation_config=gen_config)
        assistant_starts_at = len(input_ids[0])

        if self.use_transformer_heads:
            response = self.tokenizer.decode(output.sequences[0][assistant_starts_at:], skip_special_tokens=True)
            if return_hallucination_scores:
                for key in output.head_outputs:
                    if "hallu" in key:
                        probs = torch.sigmoid(output.head_outputs[key].flatten()).cpu().detach().numpy()
                        chat += [{
                            "role": "assistant",
                            "content": response,
                        }]
                        scores = self._pack_probs(
                            probs=probs,
                            input_ids=output.sequences[:, -len(probs):],
                            original_chat=chat,
                            return_assistant_only=False,
                            only_readable_tokens=True,
                        )
                        break
                else:
                    raise Exception(
                        "Failed to find the head for token level hallucination prediction."
                    )
        else:
            response = self.tokenizer.decode(output[0][assistant_starts_at:], skip_special_tokens=True)
            if return_hallucination_scores:
                chat += [{
                    "role": "assistant",
                    "content": response,
                }]
                scores = self.predict(chat=chat, token_level=True, return_assistant_only=True)

        if return_hallucination_scores:
            return response, scores

        return response


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="RAGognizer Hallucination Detector")
    parser.add_argument("repo_name", nargs="?", default=None, help="RAGognizer repository name (if not using checkpoint_dir)")
    parser.add_argument("--cpdir", default=None, help="Checkpoint directory (if not using repo_name)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--mlp", action="store_true", default=False, help="Use separate MLP instead of transformer_heads library")
    args = parser.parse_args()

    # Example usage:
    # python ragognizer/detectors/RAGognizer.py F4biian/RAGognizer-Qwen3-4B-Instruct-2507
    # or
    # python ragognizer/detectors/RAGognizer.py --cpdir /path/to/checkpoint/dir

    with torch.no_grad():
        ragognizer = RAGognizer(
            ragognizer_repo_name=args.repo_name,
            use_transformer_heads_library=not args.mlp,
            checkpoint_dir=args.cpdir,
            device=args.device,
        )

        result1 = ragognizer.predict(
            document="Albert Einstein developed the theory of relativity in the early 20th century and won the Nobel Prize in Physics in 1921.",
            user="When did Einstein develop the theory of relativity and what did he win the Nobel Prize for?",
            response="Einstein developed the theory of relativity in 1915 and won the Nobel Prize for it in 1921.",
            token_level=True,
            verbose=True,
        )

        print("\n"*5)

        result2 = ragognizer.predict(
            document="In 2018, Microsoft acquired GitHub for $7.5 billion in stock. GitHub continued to operate independently as a subsidiary.",
            user="Who acquired GitHub and for how much?",
            response="Google acquired GitHub for $7.5 billion in cash.",
            token_level=True,
            verbose=True,
        )

        response = ragognizer.generate([
            {
                "role": "user",
                "content": (
                    "Use this context to shortly answer the user's question:\n"
                    "<context>\nAlbert Einstein developed the theory of relativity in the early 20th century and won the Nobel Prize in Physics in 1921.\n</context>\n\n"
                    "<user>When did Einstein develop the theory of relativity and what did he win the Nobel Prize for?</user>"
                ),
            }
        ])
        print(response)