import argparse
import time

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from cnn.networks.lenet import LeNet

batch_size = 128
epochs = 1

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch-size", type=int, default=batch_size,
                help="(optional) batch size")
ap.add_argument("-e", "--epochs", type=int, default=epochs,
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
    data / 255.0, mnist_dataset.target.astype("int"), test_size=0.14)

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = LeNet.build(width=28, height=28, depth=1, classes=10)

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

print("[INFO] training...")
history = model.fit(trainData, trainLabels, batch_size=batch_size, nb_epoch=epochs,
                    verbose=1, validation_data=(testData, testLabels))

val_accuracy = history.history['val_acc'][-1]
val_loss = history.history['val_loss'][-1]
accuracy = history.history['acc'][-1]
loss = history.history['loss'][-1]

print("[INFO] accuracy: {:.2f}%".format(val_accuracy * 100))

end_time = time.time()
time_elapsed = end_time - start_time
print("[INFO] time: {:.2f}".format(time_elapsed))

file_results = open("./output/results_" + str(args["gpu"]) + ".txt", "a")
file_results.write(
    str(batch_size) + "," + str(epochs) + "," + str(val_loss) + "," + str(val_accuracy) + "," + str(loss) + "," + str(
        accuracy) + "," + str(time_elapsed) + "\n")
file_results.close()
