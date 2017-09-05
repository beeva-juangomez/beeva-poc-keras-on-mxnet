#!/bin/sh

python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 64 --disp-batches 50 > output/results_all_1_64_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 128 --disp-batches 50 > output/results_all_1_128_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 256 --disp-batches 50 > output/results_all_1_256_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50 > output/results_all_1_512_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 1024 --disp-batches 50 > output/results_all_1_1024_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 2048 --disp-batches 50 > output/results_all_1_2048_12.txt 2>&1
python example/image-classification/train_mnist.py --gpus 0 --num-epochs 12 --network lenet --batch-size 4096 --disp-batches 50 > output/results_all_1_4096_12.txt 2>&1
