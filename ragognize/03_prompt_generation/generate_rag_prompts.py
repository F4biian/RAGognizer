########################################################################################
# IMPORTS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
from pprint import pprint
import random
import json
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from prompt_template_randomizer import get_random_template
from FlagEmbedding import BGEM3FlagModel

emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def get_vector(text: str) -> np.ndarray:
    return emb_model.encode([text])['dense_vecs'][0]

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
SUITABLE_ARTICLES_FILE = os.path.join(DATA_DIR, "suitable_articles.json")
USER_PROMPTS_AND_ANSWERS_FILE = os.path.join(DATA_DIR, "filtered_user_prompts_and_answers.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "rag_prompts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SUITABLE_ARTICLES_FILE, "r") as file:
    suitable_articles = json.load(file)
with open(USER_PROMPTS_AND_ANSWERS_FILE, "r") as file:
    user_prompts_and_answers = json.load(file)

random.seed(432)
np.random.seed(432)

splits = {
    "train": 0.4,
    # "val": 0.1,
    "test": 0.6,
}

def get_passage_as_chunk(art, passage) -> tuple[int, int, str]:
    rand_char_before = random.randint(0, 500)
    rand_char_after = random.randint(0, 500)

    chunk_start = max(passage["start"] - rand_char_before, 0)
    chunk_end = min(passage["end"] + rand_char_after, len(art["content"]))
    chunk_str = art["content"][chunk_start:chunk_end]

    return chunk_start, chunk_end, chunk_str

def get_chunks_per_split(splits, suitable_articles, user_prompts_and_answers_splits) -> dict:
    # Map each index to, e.g., "test" or "ft"
    art_index_to_split_mapping = {}
    for s in user_prompts_and_answers_splits:
        for up in user_prompts_and_answers_splits[s]:
            art_index_to_split_mapping[up["details"]["suitable_article_index"]] = s

    splits_as_list = list(splits.keys())
    splits_as_weights = list(splits.values())

    chunks_per_split = {s: {"chunks": [], "vectors": []} for s in splits}

    pbar = tqdm(total=len(suitable_articles))
    for art_index, art in enumerate(suitable_articles):
        pbar.update()

        # Determine the split all these chunks have to be assigned to
        assign_these_chunks_to_split = art_index_to_split_mapping.get(art_index, None)
        if assign_these_chunks_to_split is None:
            assign_these_chunks_to_split = np.random.choice(splits_as_list, p=splits_as_weights)
        
        for passage_index in range(len(art["passage_data"])):
            passage = art["passage_data"][passage_index]

            _, _, chunk_str = get_passage_as_chunk(art, passage)

            # Calculate embedding
            emb_vector = get_vector(chunk_str)

            chunks_per_split[assign_these_chunks_to_split]["chunks"].append({
                "suitable_article_index": art_index,
                "suitable_article_title": art["title"],
                "passage_index": passage_index,
                "text": chunk_str
            })
            chunks_per_split[assign_these_chunks_to_split]["vectors"].append(emb_vector)

    for s in chunks_per_split:
        chunks_per_split[s]["vectors"] = np.array(chunks_per_split[s]["vectors"])

    return chunks_per_split

# Just add the index to each dict
for up_i, user_prompt in enumerate(user_prompts_and_answers):
    user_prompts_and_answers[up_i]["user_prompt_index"] = up_i

# Shuffle to remove temporal order (and thus potential temporal bias)
np.random.shuffle(user_prompts_and_answers)

# Split user prompts and their answers into the according splits
user_prompts_and_answers_splits = {}
last_end = 0
for i, s in enumerate(splits):
    perc = splits[s]
    if i >= len(splits)-1:
        user_prompts_and_answers_splits[s] = user_prompts_and_answers[last_end:]
    else:
        new_end = last_end + int(len(user_prompts_and_answers) * perc)
        user_prompts_and_answers_splits[s] = user_prompts_and_answers[last_end:new_end]
    last_end = new_end

# Create chunks and their embedding vectors using BGE-M3 (per split)
chunks_per_split = get_chunks_per_split(splits, suitable_articles, user_prompts_and_answers_splits)

# Print the amount of chunks per split (should be very similar to the original splits)
chunk_sum = 0
for s in chunks_per_split:
    chunk_sum += len(chunks_per_split[s]["chunks"])
for s in chunks_per_split:
    print(s, len(chunks_per_split[s]["chunks"]), len(chunks_per_split[s]["chunks"])/chunk_sum)


# Create the answerable and unanswerable RAG prompt for the current user prompt
def create_rag_prompt_duo(user_prompt, chunks, suitable_articles):
    user = user_prompt["user_prompt"]
    art_index = user_prompt["details"]["suitable_article_index"]
    passage_index = user_prompt["details"]["passage_index"]
    art = suitable_articles[art_index]

    chunks_n = np.random.randint(1, 6)
    source_chunk_index = np.random.randint(0, chunks_n)

    user_vector = get_vector(user)

    # Retrieve chunks_n chunks that are not associated with the suitable article of the user_prompt
    ## Compute the cosine similarities
    dot_products = np.dot(chunks["vectors"], user_vector)
    norm_vector = np.linalg.norm(user_vector)
    norms_matrix = np.linalg.norm(chunks["vectors"], axis=1)
    cosine_similarities = dot_products / (norm_vector * norms_matrix)

    chosen_articles = set()
    unanswerable_chunks = []
    sorted_chunk_indices = np.argsort(cosine_similarities)[::-1]
    for i, sorted_index in enumerate(sorted_chunk_indices):
        candidate_art_index = chunks["chunks"][sorted_index]["suitable_article_index"]
        if art_index == candidate_art_index:
            continue
        if candidate_art_index in chosen_articles:
            continue
        else:
            chosen_articles.add(candidate_art_index)
        unanswerable_chunks.append(chunks["chunks"][sorted_index])
        if len(unanswerable_chunks) >= chunks_n:
            break

    answerable_chunks = deepcopy(unanswerable_chunks)
    passage_as_chunk_start, passage_as_chunk_end, passage_as_chunk = get_passage_as_chunk(art, art["passage_data"][passage_index])
    answerable_chunks[source_chunk_index] = {"suitable_article_title": art["title"], "text": passage_as_chunk}

    info_unanswerable = "\n".join([f"#### {c['suitable_article_title']}\n{c['text']}\n\n" for c in unanswerable_chunks]).strip()
    info_answerable   = "\n".join([f"#### {c['suitable_article_title']}\n{c['text']}\n\n" for c in answerable_chunks]).strip()

    sample_log, template, settings = get_random_template()

    def replace_var(temp, var, value):
        for msg in temp:
            msg["content"] = msg["content"].replace(var, value)

    rag_prompt_answerable = deepcopy(template)
    replace_var(rag_prompt_answerable, "{{question}}", user)
    replace_var(rag_prompt_answerable, "{{info}}", info_answerable)

    rag_prompt_unanswerable = deepcopy(template)
    replace_var(rag_prompt_unanswerable, "{{question}}", user)
    replace_var(rag_prompt_unanswerable, "{{info}}", info_unanswerable)

    # print("#"*50)
    # pprint(rag_prompt_answerable, sort_dicts=False, width=500)
    # print("#"*50)
    # pprint(rag_prompt_unanswerable, sort_dicts=False, width=500)
    # print("#"*50)
    # exit()

    return {
        "user": user,
        "rag_prompt_answerable": rag_prompt_answerable,
        "answerable_answer": user_prompt["golden_answer"],
        "rag_prompt_unanswerable": rag_prompt_unanswerable,
        "unanswerable_answer": user_prompt["cluelessness_answer"],
        "details": {
            "amount_of_chunks": chunks_n,
            "unanswerable_chunks": unanswerable_chunks,
            "answerable_chunks": answerable_chunks,
            "info_unanswerable": info_unanswerable,
            "info_answerable": info_answerable,
            "source_chunk_index": source_chunk_index,
            "template": template,
            "template_details": {
                "sample_log": sample_log,
                "settings": settings,
            },
            "user_prompt": user_prompt,
        },
    }


# Now, create the RAG prompts for each split
for s in user_prompts_and_answers_splits:
    longest_prompt_len = None
    longest_prompt_index = None
    longest_prompt_answerable = None

    curr_rag_prompts = []
    for up in tqdm(user_prompts_and_answers_splits[s]):
        while True:
            entry = create_rag_prompt_duo(user_prompt=up, chunks=chunks_per_split[s], suitable_articles=suitable_articles)
            if sum(len(m["content"]) for m in entry["rag_prompt_answerable"]) < 4500:
                if sum(len(m["content"]) for m in entry["rag_prompt_unanswerable"]) < 4500:
                    break
            print(f"One of the two prompts for up {user_prompts_and_answers_splits[s].index(up)} is too long, trying to create new ones...")
        curr_rag_prompts.append(entry)

        if longest_prompt_len is None or longest_prompt_len < sum(len(m["content"]) for m in entry["rag_prompt_answerable"]):
            longest_prompt_len = sum(len(m["content"]) for m in entry["rag_prompt_answerable"])
            longest_prompt_answerable = True
            longest_prompt_index = len(curr_rag_prompts)-1
        if longest_prompt_len is None or longest_prompt_len < sum(len(m["content"]) for m in entry["rag_prompt_unanswerable"]):
            longest_prompt_len = sum(len(m["content"]) for m in entry["rag_prompt_unanswerable"])
            longest_prompt_answerable = False
            longest_prompt_index = len(curr_rag_prompts)-1

    print("Longest prompt:\n")
    if longest_prompt_answerable:
        for m in curr_rag_prompts[longest_prompt_index]["rag_prompt_answerable"]:
            print("ROLE:\n", m["role"])
            print("CONTENT:\n", m["content"])
    else:
        for m in curr_rag_prompts[longest_prompt_index]["rag_prompt_unanswerable"]:
            print("ROLE:\n", m["role"])
            print("CONTENT:\n", m["content"])
    print("Longest prompt answerable:", longest_prompt_answerable)
    print("Longest prompt index:", longest_prompt_index)
    print("Longest prompt length:", longest_prompt_len)

    np.random.shuffle(curr_rag_prompts)

    out_dir = os.path.join(OUTPUT_DIR, s)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"rag_prompts_{s}.json")
    with open(file_path, "w") as file:
        json.dump(curr_rag_prompts, file, ensure_ascii=False, indent=4)

# pprint(sample_log, width=150, sort_dicts=False)
# pprint(settings, width=150, sort_dicts=False)
# pprint(template, width=150, sort_dicts=False)