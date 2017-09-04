#!/bin/sh

python lenet_mnist.py --batch-size 64 --epochs 12 --gpu 1 > output/results_all_1_64_12.txt
python lenet_mnist.py --batch-size 128 --epochs 12 --gpu 1 > output/results_all_1_128_12.txt
python lenet_mnist.py --batch-size 256 --epochs 12 --gpu 1 > output/results_all_1_256_12.txt
python lenet_mnist.py --batch-size 512 --epochs 12 --gpu 1 > output/results_all_1_512_12.txt
python lenet_mnist.py --batch-size 1024 --epochs 12 --gpu 1 > output/results_all_1_1024_12.txt
python lenet_mnist.py --batch-size 2048 --epochs 12 --gpu 1 > output/results_all_1_2048_12.txt
python lenet_mnist.py --batch-size 4096 --epochs 12 --gpu 1 > output/results_all_1_4096_12.txt
python lenet_mnist.py --batch-size 8192 --epochs 12 --gpu 1 > output/results_all_1_8192_12.txt
