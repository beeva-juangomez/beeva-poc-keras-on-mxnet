#!/bin/sh

python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 64 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_64_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 128 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_128_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 256 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_256_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_512_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 1024 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_1024_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 2048 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_2048_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0 --num-epochs 12 --network lenet --batch-size 4096 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_1_4096_12.txt 2>&1

python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 64 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_64_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 128 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_128_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 256 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_256_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_512_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 1024 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_1024_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 2048 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_2048_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3 --num-epochs 12 --network lenet --batch-size 4096 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_4_4096_12.txt 2>&1

python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 64 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_64_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 128 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_128_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 256 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_256_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_512_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 1024 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_1024_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 2048 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_2048_12.txt 2>&1
python example/image-classification/train_mnist.py  --gpus 0,1,2,3,4,5,6,7 --num-epochs 12 --network lenet --batch-size 4096 --disp-batches 50 > output/results_mxnet_official/cudnn/results_mxnet_official_8_4096_12.txt 2>&1