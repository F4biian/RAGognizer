#!/bin/sh
set -e

python3 -m venv .venv
. ./.venv/bin/activate
pip install --upgrade pip wheel setuptools

pip install -r requirements.txt

pip install git+https://github.com/F4biian/transformers-v4.47.1-and-internal-states.git@75df852cadde684443cbf95304256381203491c2
pip install adapters==1.1.0 --no-deps
pip install transformer_heads==0.2.2 --no-deps
pip install fire==0.7.1
pip install bitsandbytes==0.49.1
pip install tabulate==0.9.0