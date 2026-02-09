from dotenv import load_dotenv, find_dotenv
import numpy as np
load_dotenv(find_dotenv())

import json
import os
from datasets import load_dataset
from datasets import Dataset, DatasetDict

import pandas as pd

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "data")
DATASET_FOLDER = os.path.join(DATA_DIR, "rag_prompts")
NEW_DATASET_FOLDER = os.path.join(DATA_DIR, "RAGognize")
os.makedirs(NEW_DATASET_FOLDER, exist_ok=True)

outputs_test_dir = os.path.join(DATA_DIR, "llm_outputs_test")
outputs_train_dir = os.path.join(DATA_DIR, "llm_outputs_train")

output_split_by_llm_by_rag_prompt_index_by_answerability = {"train": {}, "test": {}}
llms = []

def set_data(dir: str, split: str):
    global output_split_by_llm_by_rag_prompt_index_by_answerability, llms

    for llm in os.listdir(dir):
        if llm not in output_split_by_llm_by_rag_prompt_index_by_answerability:
            output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm] = {}

        if llm not in llms:
            llms.append(llm)

        output_file = os.path.join(dir, llm, "outputs_0.json")
        with open(output_file, "r") as fp:
            data = json.load(fp)

            for entry in data:
                rag_prompt_index = entry[f"{split}_rag_prompt_index"]

                if rag_prompt_index not in output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm]:
                    output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm][rag_prompt_index] = {}

                answerable = entry["answerable"]

                output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm][rag_prompt_index][answerable] = entry

set_data(outputs_train_dir, "train")
set_data(outputs_test_dir, "test")

llms = sorted(llms)

dataset = load_dataset(DATASET_FOLDER)

splits = {}

for split in dataset:
    curr_split = []
    for row_i, row in enumerate(dataset[split]):
        user = row["user"]
        details = row["details"]
        user_prompt_index = details["user_prompt"]["user_prompt_index"]
        passage_containing_answer = details["user_prompt"]["details"]["suitable_article"]["passage_data"][details["user_prompt"]["details"]["passage_index"]]
        information_asked_for = details["user_prompt"]["details"]["answer_quote_from_passage"]

        unanswerable_docs = [{"title": doc["suitable_article_title"], "text": doc["text"]} for doc in details["unanswerable_chunks"]]
        answerable_docs = [{"title": doc["suitable_article_title"], "text": doc["text"]} for doc in details["answerable_chunks"]]

        information_date = str(pd.to_datetime([passage_containing_answer["earliest_access_date"], passage_containing_answer["earliest_archive_date"], passage_containing_answer["earliest_date"]]).min())

        rag_prompt_answerable = row["rag_prompt_answerable"]
        answerable_answer = row["answerable_answer"]

        llm_responses_answerable = {}
        for llm in llms:
            entry = output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm][row_i][True]
            del entry[f"{split}_rag_prompt_index"]
            llm_responses_answerable[llm] = {
                "text": entry["output"],
                "hallucinations": entry["annotations"]["result"]["hallucinations"],
                "details": entry,
            }

        curr_split.append({
            "user_prompt_index": user_prompt_index,
            "user_prompt": user,
            "answerable": True,
            "information_type": details["user_prompt"]["information_type"],
            "category": details["user_prompt"]["category"],
            "tags": details["user_prompt"]["details"]["user_prompt_requirements"],
            "information_date": information_date,
            "documents": answerable_docs,
            "documents_str": details["info_answerable"],
            "rag_prompt": rag_prompt_answerable,
            "responses": {
                "golden_answer": answerable_answer,
                **llm_responses_answerable
            },
            "details": details
        })

        rag_prompt_unanswerable = row["rag_prompt_unanswerable"]
        unanswerable_answer = row["unanswerable_answer"]

        llm_responses_unanswerable = {}
        for llm in llms:
            entry = output_split_by_llm_by_rag_prompt_index_by_answerability[split][llm][row_i][False]
            del entry[f"{split}_rag_prompt_index"]
            llm_responses_unanswerable[llm] = {
                "text": entry["output"],
                "hallucinations": entry["annotations"]["result"]["hallucinations"],
                "details": entry,
            }

        curr_split.append({
            "user_prompt_index": user_prompt_index,
            "user_prompt": user,
            "answerable": False,
            "information_type": details["user_prompt"]["information_type"],
            "category": details["user_prompt"]["category"],
            "tags": details["user_prompt"]["details"]["user_prompt_requirements"],
            "information_date": information_date,
            "documents": unanswerable_docs,
            "documents_str": details["info_unanswerable"],
            "rag_prompt": rag_prompt_unanswerable,
            "responses": {
                "golden_answer": unanswerable_answer,
                **llm_responses_unanswerable
            },
            "details": details
        })

    # Remove entries with invalid annotations:
    new_curr_split = []
    for ie, entry in enumerate(curr_split):
        all_valid = True
        for model in entry["responses"].keys():
            if model == "golden_answer":
                continue
            all_valid &= entry["responses"][model]["details"]["annotations"]["result"]["all_valid"]

            for h in entry["responses"][model]["hallucinations"]:
                if h["start"] is None or h["end"] is None:
                    all_valid = False
                    break

        if all_valid:
            new_curr_split.append(entry)
        else:
            print("Losing entry at", ie)

    splits[split] = Dataset.from_list(new_curr_split)


dataset = DatasetDict(splits)
dataset.save_to_disk(NEW_DATASET_FOLDER)
# dataset.push_to_hub("F4biian/RAGognize")

size_per_split = {}
all_answerabilities = {}
all_infotypes = {}
all_categories = {}
all_tags = {}
all_response_level_hallus = {}

for split in dataset:
    size_per_split[split] = len(dataset[split])
    all_answerabilities[split] = pd.value_counts([entry["answerable"] for entry in dataset[split]])
    all_infotypes[split] = pd.value_counts([entry["information_type"] for entry in dataset[split]])
    all_categories[split] = pd.value_counts([entry["category"] for entry in dataset[split]])
    all_tags[split] = pd.value_counts(np.array([entry["tags"] for entry in dataset[split]]).flatten().tolist())

    llm_specific_counter = {}
    for entry in dataset[split]:
        for model in entry["responses"].keys():
            if model == "golden_answer":
                continue
            if model not in llm_specific_counter:
                llm_specific_counter[model] = 0
            if len(entry["responses"][model]["hallucinations"]) > 0:
                llm_specific_counter[model] += 1

    all_response_level_hallus[split] = pd.Series(llm_specific_counter)

size_per_split["total"] = sum(size_per_split.values())

all_answerabilities = pd.DataFrame(all_answerabilities)
all_answerabilities["total"] = all_answerabilities.sum(axis=1)
all_answerabilities.sort_values("total", ascending=False, inplace=True)

all_infotypes = pd.DataFrame(all_infotypes)
all_infotypes["total"] = all_infotypes.sum(axis=1)
all_infotypes.sort_values("total", ascending=False, inplace=True)

all_categories = pd.DataFrame(all_categories)
all_categories["total"] = all_categories.sum(axis=1)
all_categories.sort_values("total", ascending=False, inplace=True)

all_tags = pd.DataFrame(all_tags)
all_tags["total"] = all_tags.sum(axis=1)
all_tags.sort_values("total", ascending=False, inplace=True)

all_response_level_hallus = pd.DataFrame(all_response_level_hallus)
all_response_level_hallus["total"] = all_response_level_hallus.sum(axis=1)
all_response_level_hallus.sort_index(inplace=True)

# Add percentages
def add_percs(df):
    for key in df.index:
        for split in size_per_split:
            percs = df.at[key, split] / size_per_split[split]
            df.at[key, split] = f"{df.at[key, split]} ({percs*100:.1f}%)"

add_percs(all_answerabilities)
add_percs(all_infotypes)
add_percs(all_categories)
add_percs(all_tags)
add_percs(all_response_level_hallus)

print(all_answerabilities.to_markdown())
print(all_infotypes.to_markdown())
print(all_categories.to_markdown())
print(all_tags.to_markdown())
print(all_response_level_hallus.to_markdown())