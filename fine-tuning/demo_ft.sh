#!/bin/sh
set -e

# requires running `sh install.sh` before!
. ./.venv/bin/activate

# Dataset from hub:
# python fine-tuning/ft.py "Qwen/Qwen3-0.6B" F4biian/RAGognize fine-tuned_models --balanced --allentries

# Local dataset:
python ft.py "Qwen/Qwen3-0.6B" ../ragognize/data/RAGognize/ fine-tuned_models --balanced --allentries