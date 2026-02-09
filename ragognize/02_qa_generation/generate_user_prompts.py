########################################################################################
# IMPORTS

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import time
import os
from utils.shd_SimpleChatTokenLevel import SHD_SimpleChatTokenLevel
load_dotenv(find_dotenv(), override=True)

from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import random

from user_prompt_generator import UserPromptGenerator
from utils.rate_limiter import RateLimiter

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
SUITABLE_ARTICLES_FILE = os.path.join(DATA_DIR, "suitable_articles.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "user_prompts_and_answers.json")
VERBOSE = True

random.seed(432)

# Read all articles from file
suitable_articles = []
with open(SUITABLE_ARTICLES_FILE, "r") as file:
    suitable_articles = json.load(file)


google_rate_limiter = RateLimiter(
    {
        "1s": 1,
        "1m": 5,
    }
)

q_gen = UserPromptGenerator(
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", # "gemini-2.5-pro-exp-03-25"
        temperature=0.0,
        top_k=1,
        max_tokens=8000,
        timeout=300
    ),
    system_message_support=True,
    structured_output_support=True,
    rate_limiter=google_rate_limiter,
    seed=432,
)

shd = SHD_SimpleChatTokenLevel(
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", # "gemini-2.5-pro-exp-03-25"
        temperature=0.0,
        top_k=1,
        max_tokens=8000,
        timeout=300
    ),
    system_message_support=True,
    structured_output_support=True,
    rate_limiter=google_rate_limiter
)

data_points = []

last_index = -1
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        data_points = json.load(f)
        last_index = data_points[-1]["suitable_article_index"]

print("Last Index:", last_index)

pbar = tqdm(total=len(suitable_articles))

for art_index, art in enumerate(suitable_articles):
    pbar.update()
    if art_index <= last_index:
        continue

    passage_index = random.randint(0, len(art["passage_data"])-1)

    passage = art["passage_data"][passage_index]

    p_start = passage["start"]
    p_end = passage["end"]
    passage_str = art["content"][p_start:p_end]

    context_start = max(p_start - 250, 0)
    context_end = min(p_end + 250, len(art["content"]))
    context_str = art["content"][context_start:context_end]

    art_beginning = art["content"][:min(500, len(art["content"]))]

    data_point = {
        "suitable_article_index": art_index,
        "passage_index": passage_index,
        "user_prompt_generation": None,
        "golden_answer_annotation": None,
        "cluelessness_answer_annotation": None,
    }

    attempts = 3
    while attempts > 0:
        attempts -= 1
        try:
            q = q_gen.generate(art["title"], art_beginning, context_str, passage_str, verbose=VERBOSE)
            break
        except Exception as e:
            print("[User Prompt Generation] Attempt failed due to Exception:", str(e))
            time.sleep(60)
    else:
        print("[User Prompt Generation] Skipping article with index", art_index, "due to too many errors and attempts")
        continue
    
    if q is None:
        print("Skipping because q is None")
        continue

    data_point["user_prompt_generation"] = q

    if VERBOSE:
        pprint(q, indent=4, width=150, sort_dicts=False)

    if q["result"]["valid"]:
        attempts = 3
        while attempts > 0:
            attempts -= 1
            try:
                answerable_annotations = shd.annotate(
                    source_chunk={
                        "title": art["title"],
                        "chunk": context_str
                    },
                    prompt=q["result"]["user_prompt"],
                    answerable=True,
                    response=q["result"]["golden_answer"],
                    verbose=VERBOSE
                )
                data_point["golden_answer_annotation"] = answerable_annotations
                break
            except Exception as e:
                print("[Answerable SHD] Attempt failed due to Exception:", str(e))
                time.sleep(60)
        else:
            print("[Answerable SHD] Skipping article with index", art_index, "due to too many errors and attempts")
            continue
        

        if answerable_annotations is not None and answerable_annotations["result"]["all_valid"]:
            attempts = 3
            while attempts > 0:
                attempts -= 1
                try:
                    unanswerable_annotations = shd.annotate(
                        source_chunk={
                            "title": art["title"],
                            "chunk": context_str
                        },
                        prompt=q["result"]["user_prompt"],
                        answerable=False,
                        response=q["result"]["cluelessness_answer"],
                        verbose=VERBOSE
                    )
                    data_point["cluelessness_answer_annotation"] = unanswerable_annotations
                    break
                except Exception as e:
                    print("[Unanswerable SHD] Attempt failed due to Exception:", str(e))
                    time.sleep(60)
            else:
                print("[Unanswerable SHD] Skipping article with index", art_index, "due to too many errors and attempts")
                continue

    data_points.append(data_point)

    if art_index % 10 == 0:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data_points, f, ensure_ascii=False)


with open(OUTPUT_FILE, "w") as f:
    json.dump(data_points, f, ensure_ascii=False)