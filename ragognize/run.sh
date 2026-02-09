#!/bin/sh
set -e

export CUTOFF_DATE="2024-05-23 00:00:00"

# requires running `sh install.sh` before!
. ./.venv/bin/activate

# WARNING: The execution of the following scripts could take days or up to weeks depending on the amount of data and hardware used

python 00_wikipedia_scraper.py
python 01_extract_suitable_articles.py

python 02_qa_generation/generate_user_prompts.py
python 02_qa_generation/filter_user_prompts.py

python 03_prompt_generation/generate_rag_prompts.py

python 04_llm_responses/generate_outputs.py test Llama-2-7b-chat-hf "0"
python 04_llm_responses/generate_outputs.py test Llama-3.1-8B-Instruct "0"
python 04_llm_responses/generate_outputs.py test Mistral-7B-Instruct-v0.1 "0"
python 04_llm_responses/generate_outputs.py test Mistral-7B-Instruct-v0.3 "0"

python 04_llm_responses/generate_outputs.py train Llama-2-7b-chat-hf "0"
python 04_llm_responses/generate_outputs.py train Llama-3.1-8B-Instruct "0"
python 04_llm_responses/generate_outputs.py train Mistral-7B-Instruct-v0.1 "0"
python 04_llm_responses/generate_outputs.py train Mistral-7B-Instruct-v0.3 "0"

python 04_llm_responses/annotate_outputs.py test data/llm_outputs_test/
python 04_llm_responses/annotate_outputs.py train data/llm_outputs_train/

python 04_llm_responses/merge.py

# Optional:
# python 04_llm_responses/add_samples_to_ragonize.py