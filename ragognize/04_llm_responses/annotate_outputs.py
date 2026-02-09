########################################################################################
# IMPORTS

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import os
from utils.shd_SimpleChatTokenLevel import SHD_SimpleChatTokenLevel
load_dotenv(find_dotenv(), override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
import json
import time
import glob

from utils.rate_limiter import RateLimiter
from argparse import ArgumentParser

########################################################################################
parser = ArgumentParser()
parser.add_argument("split", type=str)
parser.add_argument("dir", type=str)
args = parser.parse_args()
SPLIT = args.split.strip()
dir_with_outputs = args.dir.strip()

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
RAG_PROMPT_FILE = os.path.join(DATA_DIR, "rag_prompts", SPLIT, f"rag_prompts_{SPLIT}.json")
VERBOSE = False
ANNOTATION_BATCH_SIZE = 1000

google_rate_limiter = RateLimiter(
    {
        "1s": 0.1,
        "1m": 1000,
        "1d": 10000,
    }
)

shd = SHD_SimpleChatTokenLevel(
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        top_k=1,
        max_tokens=16000,
        timeout=500
    ),
    system_message_support=True,
    structured_output_support=True,
    rate_limiter=google_rate_limiter
)


def annotate(rag_prompt_strs: list, outputs: list, max_attempts: int = 6):
    annotations = [None for _ in range(len(rag_prompt_strs))]
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            annotations = shd.annotate_batch(
                prompts=rag_prompt_strs,
                responses=outputs,
                verbose=VERBOSE
            )
            break
        except Exception as e:
            # Failed to annotate
            if attempts < max_attempts:
                time.sleep(6.65**attempts) # 6.65^6 ~= a day
    else:
        print(f"Failed completely after {max_attempts} attempts")
    return annotations

def main(outputs_file, allow_override_annotations: bool=False):
    # Load all RAG prompts
    test_rag_prompts = None
    with open(RAG_PROMPT_FILE, "r") as file:
        test_rag_prompts = json.load(file)

    # Load all LLM outputs
    data = None
    with open(outputs_file, "r") as file:
        data = json.load(file)

    curr_batch_prompts = []
    curr_batch_outputs = []
    curr_batch_i = []

    # Now, go through each output and let the SHD annotate it
    for i in tqdm(range(len(data))):
        entry = data[i]

        if not allow_override_annotations:
            if entry.get("annotations", None) is not None:
                continue

        rag_prompt_i = entry[f"{SPLIT}_rag_prompt_index"]
        rag_prompt = test_rag_prompts[rag_prompt_i]
        answerable = entry["answerable"]
        output = entry["output"]

        if answerable:
            rag_prompt = rag_prompt["rag_prompt_answerable"]
        else:
            rag_prompt = rag_prompt["rag_prompt_unanswerable"]

        rag_prompt_str = "\n\n".join(m["content"] for m in rag_prompt)

        curr_batch_prompts.append(rag_prompt_str)
        curr_batch_outputs.append(output)
        curr_batch_i.append(i)

        if len(curr_batch_outputs) > 0 and len(curr_batch_outputs) % ANNOTATION_BATCH_SIZE == 0:
            annotations = annotate(curr_batch_prompts, curr_batch_outputs)

            for data_i, anno in zip(curr_batch_i, annotations):
                data[data_i]["annotations"] = anno

            with open(outputs_file, "w") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

            curr_batch_prompts = []
            curr_batch_outputs = []
            curr_batch_i = []

    if len(curr_batch_prompts) > 0:
        annotations = annotate(curr_batch_prompts, curr_batch_outputs)
        for data_i, anno in zip(curr_batch_i, annotations):
            data[data_i]["annotations"] = anno

    with open(outputs_file, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    files = []
    for file in glob.glob(f'{dir_with_outputs.rstrip("/")}/**/*', recursive=True):
        if file.endswith(".json"):
            files.append(file)

    for f in files:
        main(f)