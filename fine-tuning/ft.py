"""
NOTE: If you get `ImportError: cannot import name 'NEED_SETUP_CACHE_CLASSES_MAPPING' from 'transformers.generation.utils'`, then please adapt the line to this:
from transformers.generation.utils import logger
NEED_SETUP_CACHE_CLASSES_MAPPING = []

Also change lines 1101 and 1133 in `.venv/lib/python3.10/site-packages/trl/trainer/sft_trainer.py` to:
`if False and not self.args.use_liger_kernel:`
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragognizer')))

import json
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd

load_dotenv(find_dotenv()) # might require "HF_TOKEN" to be set in the .env file ("HF_HOME" optionally)

from ragognizer.benchmarks.RAGTruth import RAGTruth
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig, Lfm2Model, Qwen3Model, GraniteMoeHybridModel, Gemma3TextModel
from peft import LoraConfig
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from trl import SFTTrainer, SFTConfig

from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from transformer_heads import create_headed_qlora
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.output import HeadedModelOutput
from transformer_heads.constants import model_type_map
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

from transformer_heads.constants import loss_fct_map

parser = ArgumentParser()
parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("outname")
parser.add_argument("--balanced", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--masked", action="store_true")
parser.add_argument("--allentries", action="store_true")
parser.add_argument("--ragtruth", action="store_true")
parser.add_argument("--nolmhead", action="store_true")
parser.add_argument("--headatend", action="store_true")
parser.add_argument("--quantized", action="store_true")
args = parser.parse_args()

EVAL_PERC = 0.15 # For RAGTruth
EPOCHS = 5
MODEL_NAME = args.model.strip() # "mistralai/Mistral-7B-Instruct-v0.1" # "meta-llama/Llama-3.1-8B-Instruct" # "mistralai/Mistral-7B-Instruct-v0.3" # "meta-llama/Llama-2-7b-chat-hf"
OUTNAME = args.outname.strip()
BALANCED = args.balanced
LINEAR = args.linear
MASKED = args.masked
ALL_ENTRIES = args.allentries
USE_RAGTRUTH = args.ragtruth
NO_LM_HEAD = args.nolmhead
HEAD_AT_END = args.headatend
QUANTIZED = args.quantized

PAD_TOKEN = {
    "meta-llama/Llama-2-7b-chat-hf": "<pad>",
    "mistralai/Mistral-7B-Instruct-v0.1": "<pad>",
    "mistralai/Mistral-7B-Instruct-v0.3": "<pad>",
    "LiquidAI/LFM2-1.2B-RAG": None, #"<|pad|>",
    "meta-llama/Llama-3.1-8B-Instruct": "<|reserved_special_token_0|>",
    "meta-llama/Llama-3.2-1B-Instruct": "<|reserved_special_token_0|>",
}.get(MODEL_NAME, None)

model_params = get_model_params(MODEL_NAME)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

head_perc = 0.5
layer_hook = int(model_params["num_hidden_layers"] * head_perc)
if HEAD_AT_END:
    layer_hook = 1 # Last layer
head_name = f"hallu_head_neg_{layer_hook}"

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "ragognize", "data")
OUTPUT_DIR = os.path.join(CURR_DIR, OUTNAME, MODEL_NAME.split("/")[-1])
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_type_map["qwen3"] = ("model", Qwen3Model)
model_type_map["lfm2"] = ("model", Lfm2Model)
model_type_map["granitemoehybrid"] = ("model", GraniteMoeHybridModel)
model_type_map["gemma3_text"] = ("model", Gemma3TextModel)

# if PAD_TOKEN is None:
#     raise Exception("Padding token is None! Please, set add an appropiate pad_token.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# print(tokenizer.pad_token_id)
# print(tokenizer.convert_ids_to_tokens([0])[0])

def mask_mixed_windows(arr, false_l=4, true_l=3, true_only_backward=True, min_true_group_size=3):
    arr = np.array(arr, dtype=object)  # use object type to allow np.nan assignment
    result = arr.copy()

    for i in range(len(arr)):
        if arr[i]:
            l = true_l
        else:
            l = false_l

        start = max(0, i - l)
        if true_only_backward and arr[i]:
            end = i+1
        else:
            end = min(len(arr), i + l + 1)
        window = arr[start:end]

        true_group_size = 0
        if arr[i]:
            true_group_size += 1
            for j in range(i+1, arr.shape[0]):
                if not arr[j]:
                    break
                true_group_size += 1
            for j in range(i-1, -1, -1):
                if not arr[j]:
                    break
                true_group_size += 1

        if true_group_size > 0 and true_group_size <= min_true_group_size:
            continue

        if True in window and False in window:
            result[i] = np.nan

    return result

def nan_following_true_groups(arr):
    arr = np.array(arr, dtype=object)
    result = arr.copy()

    in_first_true_group = False
    after_first_true_group = False

    for i in range(len(arr)):
        if arr[i] is True:
            if not in_first_true_group and not after_first_true_group:
                # First time seeing True → start first True group
                in_first_true_group = True
            elif after_first_true_group:
                # Already past first group, mask any future True
                result[i] = np.nan
        else:
            if in_first_true_group:
                # We finished the first True group
                in_first_true_group = False
                after_first_true_group = True
            if after_first_true_group:
                result[i] = np.nan

    return result

# Function to prepare the prompts for fine-tuning
def formatted_dataset(dataset, max_length=None):
    entries = []

    model = MODEL_NAME.split("/", 1)[1]
    
    # Go through dataset row by row
    for i, entry in enumerate(dataset):
        if max_length is not None and len(entries) >= max_length:
            break

        if "responses" not in entry:
            continue

        for curr_model in entry["responses"]:
            if curr_model == "golden_answer":
                continue

            if (curr_model == model) or ALL_ENTRIES:
                annotations = entry["responses"][curr_model]

                prompt = entry["rag_prompt"] + [
                    {
                        "role": "assistant",
                        "content": entry["responses"][curr_model]["text"]
                    }
                ]
                entry_tok = { "input_ids": tokenizer.apply_chat_template(prompt, tokenize=True) }
                entry_tok["attention_mask"] = np.ones(shape=(len(entry_tok["input_ids"]),), dtype=np.int32).tolist()

                entry_tok["labels"] = list(entry_tok["input_ids"])
                user_tokens = tokenizer.apply_chat_template(entry["rag_prompt"], tokenize=True, add_generation_prompt=True)
                assistant_token_start = len(user_tokens)

                token_starts = []
                ass_tokens = entry_tok["input_ids"][assistant_token_start:]
                last_start = 0
                last_token_found = True
                for ass_i in range(len(ass_tokens)):
                    curr_text = tokenizer.decode(ass_tokens[:ass_i+1], skip_special_tokens=True)

                    try:
                        starts_at = entry["responses"][curr_model]["text"].index(curr_text) + len(tokenizer.decode(ass_tokens[:ass_i], skip_special_tokens=True))
                        last_token_found = True
                    except:
                        starts_at = last_start
                        if last_token_found:
                            starts_at += 1
                        last_token_found = False
                        # if entry["responses"][curr_model]["text"] in curr_text:
                        #     starts_at = len(entry["responses"][curr_model]["text"])
                        # else:
                        #     print(f'#{curr_text}#')
                        #     print(f'#{entry["responses"][curr_model]["text"]}#')
                        #     print(f"curr token:", ass_tokens[ass_i], "aka", f"'{tokenizer.decode([ass_tokens[ass_i]], skip_special_tokens=False)}'")
                        #     print(f"index: {ass_i} | len: {len(ass_tokens)}")
                        #     raise Exception("Failed to verify that the cum text is part of the entire LLM response")
                        
                    token_starts.append(starts_at)
                    last_start = starts_at

                # print(token_starts)
                # print(f'#{entry["responses"][curr_model]["text"][token_starts[0]:token_starts[-1]]}#')
                # print(f'#{entry["responses"][curr_model]["text"]}#')
                # exit()

                label_per_token = np.zeros(shape=len(token_starts), dtype=bool)
                for h in annotations["hallucinations"]:
                    first_token_index_of_hallu = len(token_starts) - 1
                    try:
                        while token_starts[first_token_index_of_hallu] > h["start"]:
                            first_token_index_of_hallu -= 1
                    except:
                        first_token_index_of_hallu = 0
                    first_token_index_of_hallu = max(0, min(first_token_index_of_hallu, len(token_starts) - 1))

                    last_token_index_of_hallu = 0
                    try:
                        while token_starts[last_token_index_of_hallu] < h["end"]:
                            last_token_index_of_hallu += 1
                    except:
                        last_token_index_of_hallu = len(token_starts) - 1
                    last_token_index_of_hallu = max(0, min(last_token_index_of_hallu, len(token_starts) - 1))

                    # Set all tokens of hallucination to True
                    label_per_token[first_token_index_of_hallu:last_token_index_of_hallu] |= True
                
                if MASKED:
                    label_per_token = nan_following_true_groups(mask_mixed_windows(label_per_token, false_l=8, true_l=3, min_true_group_size=3)).astype(float)
                    label_per_token[np.isnan(label_per_token)] = -1

                # Set label of user prompt tokens to -1 (which will be masked out and ignored in loss computation)
                labels_before_response = np.full(shape=len(entry_tok["input_ids"]) - label_per_token.shape[0], fill_value=-1)
                hallu_per_token = np.concatenate([labels_before_response, label_per_token]).astype(float).tolist()
                entry_tok[head_name] = [[h] for h in hallu_per_token]
                # print(tokenizer.decode(entry_tok["labels"][assistant_token_start:], skip_special_tokens=False))

                # Ignore user tokens for language generation during training
                for label_i in range(assistant_token_start):
                    entry_tok["labels"][label_i] = -100

                entries.append(entry_tok)

                # Compare and check if still correct
                # if np.sum(label_per_token) > 0:
                #     print("#" * 50)
                #     print(annotations["hallucinations"])
                #     hids = np.array(entry_tok["input_ids"])[np.array(hallu_per_token) > 0]
                #     print(hids)
                #     print(f"%{tokenizer.batch_decode([hids], skip_special_tokens=False)[0]}%")
                #     input()

    # Create new dataset
    return Dataset.from_list(entries)

def formatted_ragtruth(get_test: bool=False, eval_perc: float=None):
    train_entries = []
    val_entries = []

    if get_test:
        eval_perc = 0.0
    else:
        if eval_perc is None:
            eval_perc = EVAL_PERC

    summ_ragtruth = RAGTruth("all", "Summary", is_test=get_test).get_entries(token_level=True)
    qa_ragtruth   = RAGTruth("all", "QA", is_test=get_test).get_entries(token_level=True)
    d2t_ragtruth  = RAGTruth("all", "Data2txt", is_test=get_test).get_entries(token_level=True)

    summ_border = int(len(summ_ragtruth) * (1 - EVAL_PERC))
    summ_ragtruth_train = summ_ragtruth[:summ_border]
    summ_ragtruth_val  = summ_ragtruth[summ_border:]

    qa_border = int(len(qa_ragtruth) * (1 - EVAL_PERC))
    qa_ragtruth_train = qa_ragtruth[:qa_border]
    qa_ragtruth_val  = qa_ragtruth[qa_border:]

    d2t_border = int(len(d2t_ragtruth) * (1 - EVAL_PERC))
    d2t_ragtruth_train = d2t_ragtruth[:d2t_border]
    d2t_ragtruth_val  = d2t_ragtruth[d2t_border:]

    train_list = summ_ragtruth_train + qa_ragtruth_train + d2t_ragtruth_train
    val_list = summ_ragtruth_val + qa_ragtruth_val + d2t_ragtruth_val

    if get_test:
        splits = [train_list]
    else:
        splits = [train_list, val_list]

    for split_index in range(len(splits)):
        for entry in splits[split_index]:
            annotations = entry["annotations"]

            prompt = entry["chat"]
            user_prompt = [msg for msg in prompt if msg["role"] != "assistant"]

            entry_tok = { "input_ids": tokenizer.apply_chat_template(prompt, tokenize=True) }
            entry_tok["attention_mask"] = np.ones(shape=(len(entry_tok["input_ids"]),), dtype=np.int32).tolist()

            entry_tok["labels"] = list(entry_tok["input_ids"])
            user_tokens = tokenizer.apply_chat_template(user_prompt, tokenize=True, add_generation_prompt=True)
            assistant_token_start = len(user_tokens)


            token_starts = []
            ass_tokens = entry_tok["input_ids"][assistant_token_start:]
            last_start = 0
            last_token_found = True
            for ass_i in range(len(ass_tokens)):
                curr_text = tokenizer.decode(ass_tokens[:ass_i+1], skip_special_tokens=True)

                try:
                    starts_at = entry["response"].index(curr_text) + len(tokenizer.decode(ass_tokens[:ass_i], skip_special_tokens=True))
                    last_token_found = True
                except:
                    starts_at = last_start
                    if last_token_found:
                        starts_at += 1
                    last_token_found = False
                    # if entry["responses"][curr_model]["text"] in curr_text:
                    #     starts_at = len(entry["responses"][curr_model]["text"])
                    # else:
                    #     print(f'#{curr_text}#')
                    #     print(f'#{entry["responses"][curr_model]["text"]}#')
                    #     print(f"curr token:", ass_tokens[ass_i], "aka", f"'{tokenizer.decode([ass_tokens[ass_i]], skip_special_tokens=False)}'")
                    #     print(f"index: {ass_i} | len: {len(ass_tokens)}")
                    #     raise Exception("Failed to verify that the cum text is part of the entire LLM response")
                    
                token_starts.append(starts_at)
                last_start = starts_at

            # print(token_starts)
            # print(f'#{entry["responses"][curr_model]["text"][token_starts[0]:token_starts[-1]]}#')
            # print(f'#{entry["responses"][curr_model]["text"]}#')
            # exit()

            label_per_token = np.zeros(shape=len(token_starts), dtype=bool)
            for h in annotations:
                if h["label"] == 0:
                    continue

                first_token_index_of_hallu = len(token_starts) - 1
                try:
                    while token_starts[first_token_index_of_hallu] > h["start"]:
                        first_token_index_of_hallu -= 1
                except:
                    first_token_index_of_hallu = 0
                first_token_index_of_hallu = max(0, min(first_token_index_of_hallu, len(token_starts) - 1))

                last_token_index_of_hallu = 0
                try:
                    while token_starts[last_token_index_of_hallu] < h["end"]:
                        last_token_index_of_hallu += 1
                except:
                    last_token_index_of_hallu = len(token_starts) - 1
                last_token_index_of_hallu = max(0, min(last_token_index_of_hallu, len(token_starts) - 1))

                # Set all tokens of hallucination to True
                label_per_token[first_token_index_of_hallu:last_token_index_of_hallu] |= True
            
            if MASKED:
                label_per_token = nan_following_true_groups(mask_mixed_windows(label_per_token, false_l=8, true_l=3, min_true_group_size=3)).astype(float)
                label_per_token[np.isnan(label_per_token)] = -1

            # Set label of user prompt tokens to -1 (which will be masked out and ignored in loss computation)
            labels_before_response = np.full(shape=len(entry_tok["input_ids"]) - label_per_token.shape[0], fill_value=-1)
            hallu_per_token = np.concatenate([labels_before_response, label_per_token]).astype(float).tolist()
            entry_tok[head_name] = [[h] for h in hallu_per_token]
            # print(tokenizer.decode(entry_tok["labels"][assistant_token_start:], skip_special_tokens=False))

            # Ignore user tokens for language generation during training
            for label_i in range(assistant_token_start):
                entry_tok["labels"][label_i] = -100

            if split_index == 0:
                train_entries.append(entry_tok)
            else:
                val_entries.append(entry_tok)

            # Compare and check if still correct
            # if np.sum(label_per_token) > 0:
            #     print("#" * 50)
            #     print(annotations)
            #     hids = np.array(entry_tok["input_ids"])[np.array(hallu_per_token) > 0]
            #     print(hids)
            #     print(f"%{tokenizer.batch_decode([hids], skip_special_tokens=False)[0]}%")
            #     input()

    # Create new datasets
    if get_test:
        Dataset.from_list(train_entries)
    else:
        return Dataset.from_list(train_entries), Dataset.from_list(val_entries)

if USE_RAGTRUTH:
    train_dataset, val_dataset = formatted_ragtruth()
    test_dataset = formatted_ragtruth(get_test=True)
else:
    dataset_path = args.dataset
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # dataset = load_dataset("F4biian/RAGognize")
        dataset = load_dataset(dataset_path)
    split_dataset = formatted_dataset(dataset["train"])
    split_dataset = split_dataset.train_test_split(
        test_size=0.1,
        seed=42
    )
    train_dataset = split_dataset["train"]
    val_dataset   = split_dataset["test"]
    test_dataset  = formatted_dataset(dataset["test"])

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", head_name, "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", head_name, "labels"],
)
test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", head_name, "labels"],
)

# Calculcate weight for hallucinated tokens (to account for imbalance)
if BALANCED:
    zeros = 0
    ones = 0
    for ten in train_dataset[head_name]:
        zeros += (ten == 0).sum().item()
        ones += (ten == 1).sum().item()

    ones_weight = 1 / (ones / (zeros + ones))
else:
    ones_weight = 1.0

print("Zeros:", zeros)
print(" Ones:", ones)
print("Total:", ones + zeros)
print("Ones Weight:", ones_weight)

print(train_dataset)
print(val_dataset)
print(test_dataset)

class Masked_BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Mask out entries where target < -0.1 (should be -1 and -0.1 is for safety)
        mask = target >= -0.1

        # Avoid computing loss if all entries are masked out
        if not mask.any():
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        return super().forward(input[mask], target[mask])

# Add custom loss to transformer heads lib
loss_fct_map["masked_bce_with_logits"] = Masked_BCEWithLogitsLoss(pos_weight=torch.tensor([ones_weight]).to("cuda"))


if NO_LM_HEAD:
    head_configs = [
        HeadConfig(
            name=head_name,
            layer_hook=-layer_hook,
            in_size=hidden_size,
            hidden_size=1024,
            num_layers=1 if LINEAR else 3,
            output_activation="linear",
            pred_for_sequence=False,
            loss_fct="masked_bce_with_logits",
            num_outputs=1,
        )
    ]
else:
    head_configs = [
        HeadConfig(
            name=head_name,
            layer_hook=-layer_hook,
            in_size=hidden_size,
            hidden_size=1024,
            num_layers=1 if LINEAR else 3,
            output_activation="linear",
            pred_for_sequence=False,
            loss_fct="masked_bce_with_logits",
            num_outputs=1,
        ),
        HeadConfig(
            name="lm_head",
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=True,
            pred_for_sequence=False,
            loss_fct="cross_entropy",
            num_outputs=vocab_size,
            is_regression=False,
            trainable=False,
        )
    ]

quantization_config = None
if QUANTIZED:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = create_headed_qlora(
    base_model_class=model_class,
    model_name=MODEL_NAME,
    quantization_config=quantization_config,
    lora_config=lora_config,
    head_configs=head_configs,
    fully_trained_heads=True,
    device_map={"": torch.cuda.current_device()},
)

print_trainable_parameters(model)

# Add pad_token to tokenizer
if PAD_TOKEN:
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        if "reserved" not in PAD_TOKEN:
            model.resize_token_embeddings(len(tokenizer))

collator = DataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }
)


# Config for fine-tuning
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=4e-5,
    num_train_epochs=1,
    packing=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    save_strategy="epoch",
    save_steps=99,
    max_length=1024,
    logging_strategy="epoch",
    logging_steps=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator
)

# test_dataset = trainer._prepare_dataset_for_kl_loss(test_dataset)

def eval(is_val: bool=True):
    model.eval()
    if is_val:
        loader = DataLoader(val_dataset, batch_size=1, collate_fn=collator)
    else:
        loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)

    all_probs  = []
    all_labels = []
    losses = []

    for bi, batch in tqdm(
        enumerate(loader), total=len(loader), desc="Evaluating"
    ):
        try:
            outputs: HeadedModelOutput = model(
                **{key: value.to(model.device) for key, value in batch.items()}
            )
            losses.append(float(outputs.loss.item()))

            preds_per_token = outputs.preds_by_head[head_name].flatten()
            
            probs = torch.sigmoid(preds_per_token).cpu().detach().numpy()
            labels = batch[head_name][0, :].cpu().detach().numpy().flatten()

            # Mask out user prompt token (having -1 label)
            mask = labels >= -0.1

            all_probs.extend(list(probs[mask]))
            all_labels.extend(list(labels[mask]))

            del outputs, preds_per_token
        except torch.OutOfMemoryError:
            print(f"OOM Error for batch {bi}")
        del batch

    torch.cuda.empty_cache()

    loss = float(np.mean(losses))

    all_probs_np = np.array(all_probs).flatten()
    all_labels_np = np.array(all_labels, dtype=int).flatten()
    total_roc_auc = roc_auc_score(all_labels_np, all_probs_np)
    total_pr_auc = average_precision_score(all_labels_np, all_probs_np)

    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    j_statistic = tpr - fpr
    best_threshold_index = np.argmax(j_statistic)
    best_threshold = thresholds[best_threshold_index]

    result = {
        "loss": loss,
        "roc_auc": float(total_roc_auc),
        "pr_auc": float(total_pr_auc),
        "best_threshold": float(best_threshold),
    }

    return result

training_loss_history = pd.Series()
eval_loss_history  = pd.Series()
eval_auroc_history = pd.Series()
eval_auprc_history = pd.Series()

print("Pre-Training Evaluation")
eval_res = eval()
print("evaluation:", eval_res)
eval_loss_history[0] = eval_res["loss"]
eval_auroc_history[0] = eval_res["roc_auc"]
eval_auprc_history[0] = eval_res["pr_auc"]

for epoch in range(EPOCHS):
    model.train()
    print("EPOCH:", epoch+1)
    out = trainer.train()
    print("training: ", out)
    
    torch.cuda.empty_cache()

    training_loss_history[epoch+1] = out.training_loss

    eval_res = eval()
    print("evaluation:", eval_res)

    eval_loss_history[epoch+1] = eval_res["loss"]
    eval_auroc_history[epoch+1] = eval_res["roc_auc"]
    eval_auprc_history[epoch+1] = eval_res["pr_auc"]

    save_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{epoch+1}")
    print("Saving to", save_dir)

    model.save_pretrained(save_dir)

test_res = eval(is_val=False)
print("test:", test_res)

print("training_loss_history")
print(training_loss_history)
print("eval_loss_history")
print(eval_loss_history)
print("eval_auroc_history")
print(eval_auroc_history)
print("eval_auprc_history")
print(eval_auprc_history)

plot_dir = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Save metrics
with open(os.path.join(plot_dir, "metrics.json"), "w") as file:
    json.dump({
        "training_loss_history": training_loss_history.to_dict(),
        "eval_loss_history": eval_loss_history.to_dict(),
        "eval_auroc_history": eval_auroc_history.to_dict(),
        "eval_auprc_history": eval_auprc_history.to_dict(),
        "test": test_res,
    }, file, ensure_ascii=False, indent=4)


# Save plot
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(training_loss_history.index, training_loss_history.values, label='Training Loss', color='blue')
ax1.plot(eval_loss_history.index, eval_loss_history.values, label='Eval Loss', color='orange')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.plot(eval_auroc_history.index, eval_auroc_history.values, label='Eval AUROC', color='green')
ax2.plot(eval_auroc_history.index, eval_auprc_history.values, label='Eval AUPRC', color='red')
ax2.set_ylabel("AUROC / AUPRC")
ax2.tick_params(axis='y')
plt.title("Training & Evaluation Loss and Evaluation AUROC / AUPRC Over Epochs")
fig.tight_layout()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
ax1.grid(True)
plt.savefig(os.path.join(plot_dir, "loss_and_aucs.svg"), format='svg')
plt.close()