#!/usr/bin/env bash
source activate qb
pip install gensim
python -m qanta.dan2 web
