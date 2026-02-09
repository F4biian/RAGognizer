# Usage examples:
# python data_creation/stage_3_llm_outputs/generate_outputs.py Llama-2-7b-chat-hf "0" --eis --max 1000 --datadir /custom/data/dir/
# python data_creation/stage_3_llm_outputs/generate_outputs.py Llama-2-7b-chat-hf "59" --checkpoint "checkpoint-59" --datadir /custom/data/dir/
# python data_creation/stage_3_llm_outputs/generate_outputs.py Mistral-7B-Instruct-v0.3 "0" --datadir /custom/data/dir/
# python data_creation/stage_3_llm_outputs/generate_outputs.py Llama-2-7b-chat-hf "0" --eis --datadir /custom/data/dir/ --layers "[0.375,0.4375,0.5,0.5625]"
import os

########################################################################################
# IMPORTS
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

# HF_HOME may be set in the .env file (optional)
# HF_TOKEN has to be set in the .env file (to access certain LLMs)

from utils import Llama2_7B_Chat_HF, Llama3_1_8B_Instruct, Mistral_7B_Instruct_v0_1, Mistral_7B_Instruct_v0_3, LLM
import json
import h5py
from tqdm import tqdm
from argparse import ArgumentParser

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
RAG_PROMPT_FILE = os.path.join(DATA_DIR, "rag_prompts", "SPLIT", "rag_prompts_SPLIT.json")
FT_DIR = os.path.join(DATA_DIR, "..", "ft")

def load_rag_prompts(split: str):
    with open(RAG_PROMPT_FILE.replace("SPLIT", split), "r") as f:
        return json.load(f)

def generate_outputs(llm: LLM, suffix: str, split: str, different_data_dir=None, max_prompts: int=None, extract_internal_states: bool=False, layers: str="all"):
    if different_data_dir is None:
        output_dir = os.path.join(DATA_DIR, f"llm_outputs_{split}", llm.name.split("/")[-1])
    else:
        output_dir = os.path.join(different_data_dir, f"llm_outputs_{split}", llm.name.split("/")[-1])

    os.makedirs(output_dir, exist_ok=True)

    # Determine what layers to extract
    quantiles = None
    if layers.strip().lower() != "all":
        quantiles = json.loads(layers)

    rag_prompts = load_rag_prompts(split)

    if max_prompts is not None:
        pbar = tqdm(total=min(max_prompts//2, len(rag_prompts)))
    else:
        pbar = tqdm(total=len(rag_prompts))

    llm.load()

    if extract_internal_states:
        cevs_file = os.path.join(output_dir, f"cevs{suffix}.h5")
        iavs_file = os.path.join(output_dir, f"iavs{suffix}.h5")

        # Delete files if they already exist
        if os.path.exists(cevs_file):
            os.remove(cevs_file)
        if os.path.exists(iavs_file):
            os.remove(iavs_file)

    data = []

    if extract_internal_states:
        with h5py.File(cevs_file, "a") as hf_cevs:
            with h5py.File(iavs_file, "a") as hf_iavs:
                for i, rp_pair in enumerate(rag_prompts):
                    # Give answerable prompt to LLM
                    group = f"{split}_answerable_{i}"
                    p = rp_pair["rag_prompt_answerable"]
                    sys_msg = None
                    prompt = None
                    for m in p:
                        if m["role"] == "system":
                            sys_msg = m["content"]
                        elif m["role"] == "user":
                            prompt = m["content"]

                    try:
                        full_prompt_str, full_chat_str, output_str, token_starts, cevs, iavs = llm.generate(prompt, system=sys_msg, internal_states_device="cpu")
                        if quantiles is not None:
                            cev_mask = []
                            iav_mask = []
                            num_layers = cevs.shape[1] - 1
                            for q in quantiles:
                                layer_i = round(num_layers * q)
                                cev_mask.append(layer_i + 1)
                                iav_mask.append(layer_i)
                            cevs = cevs[:, cev_mask, :]
                            iavs = iavs[:, iav_mask, :]
                    except Exception as e:
                        print("Error:", e)
                        continue

                    # Give unanswerable prompt to LLM
                    group_un = f"{split}_unanswerable_{i}"
                    p = rp_pair["rag_prompt_unanswerable"]
                    sys_msg = None
                    prompt = None
                    for m in p:
                        if m["role"] == "system":
                            sys_msg = m["content"]
                        elif m["role"] == "user":
                            prompt = m["content"]

                    try:
                        full_prompt_str_un, full_chat_str_un, output_str_un, token_starts_un, cevs_un, iavs_un = llm.generate(prompt, system=sys_msg, internal_states_device="cpu")
                        if quantiles is not None:
                            cev_mask = []
                            iav_mask = []
                            num_layers = cevs_un.shape[1] - 1
                            for q in quantiles:
                                layer_i = round(num_layers * q)
                                cev_mask.append(layer_i + 1)
                                iav_mask.append(layer_i)
                            cevs_un = cevs_un[:, cev_mask, :]
                            iavs_un = iavs_un[:, iav_mask, :]
                    except Exception as e:
                        print("Error:", e)
                        continue
                    
                    # Store data
                    data.append({
                        "group": group,
                        "answerable": True,
                        f"{split}_rag_prompt_index": i,
                        "full_prompt": full_prompt_str,
                        "full_chat": full_chat_str,
                        "output": output_str,
                        "token_starts": token_starts,
                    })
                    data.append({
                        "group": group_un,
                        "answerable": False,
                        f"{split}_rag_prompt_index": i,
                        "full_prompt": full_prompt_str_un,
                        "full_chat": full_chat_str_un,
                        "output": output_str_un,
                        "token_starts": token_starts_un,
                    })
                    hf_cevs.create_group(group).create_dataset("is", data=cevs.numpy(), compression="gzip", compression_opts=9)
                    hf_iavs.create_group(group).create_dataset("is", data=iavs.numpy(), compression="gzip", compression_opts=9)
                    hf_cevs.create_group(group_un).create_dataset("is", data=cevs_un.numpy(), compression="gzip", compression_opts=9)
                    hf_iavs.create_group(group_un).create_dataset("is", data=iavs_un.numpy(), compression="gzip", compression_opts=9)

                    pbar.update()

                    if max_prompts is not None:
                        if len(data) >= max_prompts:
                            break
    else:
        for i, rp_pair in enumerate(rag_prompts):
            # Give answerable prompt to LLM
            group = f"{split}_answerable_{i}"
            p = rp_pair["rag_prompt_answerable"]
            sys_msg = None
            prompt = None
            for m in p:
                if m["role"] == "system":
                    sys_msg = m["content"]
                elif m["role"] == "user":
                    prompt = m["content"]

            try:
                full_prompt_str, full_chat_str, output_str, token_starts = llm.generate(prompt, system=sys_msg)
            except Exception as e:
                print("Error:", e)
                continue

            # Give unanswerable prompt to LLM
            group_un = f"{split}_unanswerable_{i}"
            p = rp_pair["rag_prompt_unanswerable"]
            sys_msg = None
            prompt = None
            for m in p:
                if m["role"] == "system":
                    sys_msg = m["content"]
                elif m["role"] == "user":
                    prompt = m["content"]

            try:
                full_prompt_str_un, full_chat_str_un, output_str_un, token_starts_un = llm.generate(prompt, system=sys_msg)
            except Exception as e:
                print("Error:", e)
                continue
            

            # Store data
            data.append({
                "group": group,
                "answerable": True,
                f"{split}_rag_prompt_index": i,
                "full_prompt": full_prompt_str,
                "full_chat": full_chat_str,
                "output": output_str,
                "token_starts": token_starts,
            })
            data.append({
                "group": group_un,
                "answerable": False,
                f"{split}_rag_prompt_index": i,
                "full_prompt": full_prompt_str_un,
                "full_chat": full_chat_str_un,
                "output": output_str_un,
                "token_starts": token_starts_un,
            })

            pbar.update()

            if max_prompts is not None:
                if len(data) >= max_prompts:
                    break

    with open(os.path.join(output_dir, f"outputs{suffix}.json"), "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    llm.unload()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("model")
    parser.add_argument("suffix", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--eis", action="store_true", default=False)
    parser.add_argument("--max", default=None, type=int)
    parser.add_argument("--datadir", default=None, type=str)
    parser.add_argument("--layers", default="all", type=str)

    args = parser.parse_args()

    model_name = args.model.strip()
    split = args.split.strip()
    suffix = args.suffix.strip()
    checkpoint = args.checkpoint
    extract_internal_states = args.eis
    max_prompts = args.max
    datadir = args.datadir
    layers = args.layers

    llm_class = {
        "Llama-2-7b-chat-hf": Llama2_7B_Chat_HF,
        "Llama-3.1-8B-Instruct": Llama3_1_8B_Instruct,
        "Mistral-7B-Instruct-v0.3": Mistral_7B_Instruct_v0_3,
        "Mistral-7B-Instruct-v0.1": Mistral_7B_Instruct_v0_1,
    }.get(model_name, None)

    if llm_class is None:
        raise Exception("Unknown LLM: " + str(model_name))

    if checkpoint is not None and checkpoint.strip() != "":
        llm = llm_class(ft_checkpoint_dir=os.path.join(FT_DIR, model_name, checkpoint), load_with_heads=True)
    else:
        llm = llm_class()

    generate_outputs(llm, suffix=f"_{suffix}", split=split, different_data_dir=datadir, max_prompts=max_prompts, extract_internal_states=extract_internal_states, layers=layers)