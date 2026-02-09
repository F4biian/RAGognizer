########################################################################################
# IMPORTS

import os
import numpy as np
import pandas as pd

import json

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
SUITABLE_ARTICLES_FILE = os.path.join(DATA_DIR, "suitable_articles.json")
USER_PROMPTS_FILE = os.path.join(DATA_DIR, "user_prompts_and_answers.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "filtered_user_prompts_and_answers.json")

# Read all articles from file
suitable_articles = []
with open(SUITABLE_ARTICLES_FILE, "r") as file:
    suitable_articles = json.load(file)

with open(USER_PROMPTS_FILE, "r") as file:
    data = json.load(file)

entirely_valid_entries = []

cluelessness_correct_list = []
invalid_before = []
valid_list = []
got_hallucination = []

not_addressed_user_prompt = []

categories = []
information_types = []
prompt_requirements = []

for entry in data:
    suitable_article_index = entry["suitable_article_index"]
    passage_index = entry["passage_index"]

    user_prompt_generation = entry["user_prompt_generation"]
    golden_answer_annotation = entry["golden_answer_annotation"]
    cluelessness_answer_annotation = entry["cluelessness_answer_annotation"]

    if user_prompt_generation is None:
        invalid_before.append(1)
        continue
    if golden_answer_annotation is None:
        invalid_before.append(1)
        continue
    if cluelessness_answer_annotation is None:
        invalid_before.append(1)
        continue
    invalid_before.append(0)

    if not user_prompt_generation["result"]["valid"]:
        valid_list.append(0)
        continue
    if not golden_answer_annotation["result"]["all_valid"]:
        valid_list.append(0)
        continue
    if not cluelessness_answer_annotation["result"]["all_valid"]:
        valid_list.append(0)
        continue
    valid_list.append(1)

    if golden_answer_annotation["result"]["cluelessness"] is not False:
        cluelessness_correct_list.append(0)
        continue
    if cluelessness_answer_annotation["result"]["cluelessness"] is not True:
        cluelessness_correct_list.append(0)
        continue
    cluelessness_correct_list.append(1)

    if not golden_answer_annotation["result"]["addressed_user_prompt"]:
        not_addressed_user_prompt.append(1)
        continue
    if not cluelessness_answer_annotation["result"]["addressed_user_prompt"]:
        not_addressed_user_prompt.append(1)
        continue
    not_addressed_user_prompt.append(0)

    if len(golden_answer_annotation["result"]["hallucinations"]) > 0 or golden_answer_annotation["result"]["completely_hallucinated"]:
        got_hallucination.append(1)
        continue
    if len(cluelessness_answer_annotation["result"]["hallucinations"]) > 0 or golden_answer_annotation["result"]["completely_hallucinated"]:
        got_hallucination.append(1)
        continue
    got_hallucination.append(0)

    information_types.append(user_prompt_generation["result"]["information_type"])
    categories.append(user_prompt_generation["result"]["category"])
    prompt_requirements.extend(user_prompt_generation["result"]["sampled_requirements"])

    art = suitable_articles[suitable_article_index]
    passage = art["passage_data"][passage_index]

    entirely_valid_entries.append({
        "user_prompt": user_prompt_generation["result"]["user_prompt"],
        "category": user_prompt_generation["result"]["category"],
        "information_type": user_prompt_generation["result"]["information_type"],
        "article_title": art["title"],
        "passage_containing_answer": art["content"][passage["start"]:passage["end"]],
        "golden_answer": user_prompt_generation["result"]["golden_answer"],
        "cluelessness_answer": user_prompt_generation["result"]["cluelessness_answer"],
        "details": {
            "suitable_article_index": suitable_article_index,
            "suitable_article": art,
            "passage_index": passage_index,
            "user_prompt_requirements": user_prompt_generation["result"]["sampled_requirements"],
            "answer_quote_from_passage": user_prompt_generation["result"]["answer_quote"],
        }
    })

    # pprint(entry, width=150, sort_dicts=False)
    # exit()

with open(OUTPUT_FILE, "w") as file:
    json.dump(entirely_valid_entries, file, ensure_ascii=False, indent=4)

print("Prefitlered invalid entries:", np.sum(invalid_before), "/", len(invalid_before), " # mean:", np.mean(invalid_before))
print("Valid entries:", np.sum(valid_list), "/", len(valid_list), " # mean:", np.mean(valid_list))
print("Correct cluelessness:", np.sum(cluelessness_correct_list), "/", len(cluelessness_correct_list), " # mean:", np.mean(cluelessness_correct_list))
print("Entries not addressing user prompt:", np.sum(not_addressed_user_prompt), "/", len(not_addressed_user_prompt), " # mean:", np.mean(not_addressed_user_prompt))
print("Entries with hallucinations:", np.sum(got_hallucination), "/", len(got_hallucination), " # mean:", np.mean(got_hallucination))

print("#"*50)
print("Information Types Distribution (Abs):\n", pd.Series(information_types).value_counts())
print("Information Types Distribution:\n", pd.Series(information_types).value_counts()/len(information_types))

print("#"*50)
print("Categories Distribution (Abs):\n", pd.Series(categories).value_counts())
print("Categories Distribution:\n", pd.Series(categories).value_counts()/len(categories))

print("#"*50)
print("Prompt Requirements (Abs):\n", pd.Series(prompt_requirements).value_counts())
print("Prompt Requirements:\n", pd.Series(prompt_requirements).value_counts()/len(prompt_requirements))