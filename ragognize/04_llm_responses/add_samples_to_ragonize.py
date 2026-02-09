import os
import shutil
import traceback

########################################################################################
# IMPORTS
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

# HF_HOME may be set in the .env file (optional)
# HF_TOKEN has to be set in the .env file (to access certain LLMs)

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from utils import Llama2_7B_Chat_HF, Llama3_1_8B_Instruct, Mistral_7B_Instruct_v0_1, Mistral_7B_Instruct_v0_3, LLM
from tqdm import tqdm

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
SAVE_DIR = os.path.join(DATA_DIR, "RAGognize-with-samples-test")
NEW_DATASET_FOLDER = os.path.join(DATA_DIR, "RAGognize")

def load_ragognize():
    if os.path.exists(SAVE_DIR):
        dataset = load_from_disk(SAVE_DIR)
    else:
        dataset = load_from_disk(NEW_DATASET_FOLDER)
    return dataset

def safe_save(dataset: DatasetDict, save_dir: str):
    tmp_dir = save_dir + "_tmp"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    dataset.save_to_disk(tmp_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.rename(tmp_dir, save_dir)

def generate_outputs():
    ragognize: DatasetDict = load_ragognize()
    print(ragognize)

    llm_classes = {
        "Llama-2-7b-chat-hf": Llama2_7B_Chat_HF,
        "Llama-3.1-8B-Instruct": Llama3_1_8B_Instruct,
        "Mistral-7B-Instruct-v0.3": Mistral_7B_Instruct_v0_3,
        "Mistral-7B-Instruct-v0.1": Mistral_7B_Instruct_v0_1,
    }

    # pbar = tqdm(total=4*len(ragognize["train"]) + 4*len(ragognize["test"]))
    pbar = tqdm(total=4*len(ragognize["test"]))

    for llm_name in llm_classes:
        curr_llm: LLM = llm_classes[llm_name]()

        curr_llm.load()

        for split in ragognize:
            if split == "train":
                continue

            print(f"\nProcessing split={split}, model={llm_name}")
            rows = [ragognize[split][i] for i in range(len(ragognize[split]))]
            updated = False

            for i, row in enumerate(rows):
                responses = row.get("responses", {})

                for resp_llm_name in responses:
                    if resp_llm_name.lower().strip() != llm_name.lower().strip():
                        continue

                    if responses[resp_llm_name].get("samples") is not None:
                        pbar.update()
                        continue

                    samples = []

                    # extract user / system
                    user = None
                    system = None
                    for msg in row["rag_prompt"]:
                        role = msg["role"].lower().strip()
                        if role == "user":
                            user = msg["content"]
                        elif role == "system":
                            system = msg["content"]

                    if user is None:
                        print(f"WARNING: user is None at {split}[{i}] ({resp_llm_name})")
                        pbar.update()
                        continue

                    for _ in range(5):
                        try:
                            full_prompt, full_chat, response, token_starts = curr_llm.generate(
                                prompt=user,
                                system=system,
                                temperature=0.7,
                                do_sample=True,
                            )

                            samples.append({
                                "temperature": 0.7,
                                "output": response,
                                "token_starts": list(token_starts),
                            })

                        except Exception:
                            print(f"Generation error at {split}[{i}]:")
                            print(traceback.format_exc())

                    rows[i]["responses"][resp_llm_name]["samples"] = samples
                    updated = True
                    pbar.update()

                # periodic checkpoint (cheap now)
                if updated and i > 0 and i % 200 == 0:
                    ragognize[split] = Dataset.from_list(rows)
                    safe_save(ragognize, SAVE_DIR)
                    updated = False

            ragognize[split] = Dataset.from_list(rows)
            safe_save(ragognize, SAVE_DIR)

        curr_llm.unload()

    pbar.close()


if __name__ == "__main__":
    generate_outputs()