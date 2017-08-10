import argparse
import time

import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from cnn.networks.lenet import LeNet

batch_size = 128
epochs = 20

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
ap.add_argument("-b", "--batch-size", type=int, default=128,
                help="(optional) batch size")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="(optional) number of epochs")
ap.add_argument("-g", "--gpu", type=int, default=-1,
                help="(optional) whether or not model should run on GPU")
args = vars(ap.parse_args())

print("[INFO] downloading MNIST...")
mnist_dataset = datasets.fetch_mldata("MNIST Original")

start_time = time.time()

data = mnist_dataset.data.reshape((mnist_dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, mnist_dataset.target.astype("int"), test_size=0.33)

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)

if args["gpu"] > 0:
    gpus = ["gpu(" + str(index) + ")" for index in range(args["gpu"])]
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"], context=gpus)
else:
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

if args["batch_size"] > 0:
    batch_size = args["batch_size"]

if args["epochs"] > 0:
    epochs = args["epochs"]

if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=batch_size, nb_epoch=epochs,
              verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=batch_size, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    end_time = time.time()
    time_elapsed = end_time - start_time
    print("[INFO] time: {:.2f}".format(time_elapsed))

    file_results = open("./output/results.txt", "a")
    file_results.write(
        str(batch_size) + "," + str(epochs) + "," + str(loss) + "," + str(accuracy) + "," + str(time_elapsed) + "\n")
    file_results.close()

if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)
