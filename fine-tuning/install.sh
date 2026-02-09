#!/bin/sh
set -e

python3 -m venv .venv
. ./.venv/bin/activate
pip install --upgrade pip wheel setuptools

pip install -r requirements.txt