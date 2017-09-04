#!/bin/sh

python rnn_imdb.py --batch-size 64 --epochs 5 --gpu 1 > output/results_all_1_64_5.txt
python rnn_imdb.py --batch-size 64 --epochs 10 --gpu 1 > output/results_all_1_64_10.txt
python rnn_imdb.py --batch-size 64 --epochs 20 --gpu 1 > output/results_all_1_64_20.txt
python rnn_imdb.py --batch-size 128 --epochs 5 --gpu 1 > output/results_all_1_128_5.txt
python rnn_imdb.py --batch-size 128 --epochs 10 --gpu 1 > output/results_all_1_128_10.txt
python rnn_imdb.py --batch-size 128 --epochs 20 --gpu 1 > output/results_all_1_128_20.txt
python rnn_imdb.py --batch-size 256 --epochs 5 --gpu 1 > output/results_all_1_256_5.txt
python rnn_imdb.py --batch-size 256 --epochs 10 --gpu 1 > output/results_all_1_256_10.txt
python rnn_imdb.py --batch-size 256 --epochs 20 --gpu 1 > output/results_all_1_256_20.txt
python rnn_imdb.py --batch-size 512 --epochs 5 --gpu 1 > output/results_all_1_512_5.txt
python rnn_imdb.py --batch-size 512 --epochs 10 --gpu 1 > output/results_all_1_512_10.txt
python rnn_imdb.py --batch-size 512 --epochs 20 --gpu 1 > output/results_all_1_512_20.txt
python rnn_imdb.py --batch-size 1024 --epochs 5 --gpu 1 > output/results_all_1_1024_5.txt
python rnn_imdb.py --batch-size 1024 --epochs 10 --gpu 1 > output/results_all_1_1024_10.txt
python rnn_imdb.py --batch-size 1024 --epochs 20 --gpu 1 > output/results_all_1_1024_20.txt