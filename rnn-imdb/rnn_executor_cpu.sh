#!/bin/sh

python rnn_imdb.py --batch-size 64 --epochs 5
python rnn_imdb.py --batch-size 64 --epochs 10
python rnn_imdb.py --batch-size 128 --epochs 5
python rnn_imdb.py --batch-size 128 --epochs 10
python rnn_imdb.py --batch-size 256 --epochs 5
python rnn_imdb.py --batch-size 256 --epochs 10
python rnn_imdb.py --batch-size 512 --epochs 5
python rnn_imdb.py --batch-size 512 --epochs 10