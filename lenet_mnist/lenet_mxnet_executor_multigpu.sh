#!/bin/sh

python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 64 --disp-batches 50 > output/results_all_8_64_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 128 --disp-batches 50 > output/results_all_8_128_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 256 --disp-batches 50 > output/results_all_8_256_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50 > output/results_all_8_512_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 1024 --disp-batches 50 > output/results_all_8_1024_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 2048 --disp-batches 50 > output/results_all_8_2048_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 4096 --disp-batches 50 > output/results_all_8_4096_12.txt 2>&1
