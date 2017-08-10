import argparse
import time

from rnn.networks.rnn import IMDB
from keras.datasets import imdb
from keras.preprocessing import sequence

batch_size = 128
epochs = 20
top_words = 5000
max_review_length = 500
embedding_vector_length = 32

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch-size", type=int, default=batch_size,
                help="(optional) batch size")
ap.add_argument("-e", "--epochs", type=int, default=epochs,
                help="(optional) number of epochs")
ap.add_argument("-g", "--gpu", type=int, default=-1,
                help="(optional) whether or not model should run on GPU")
args = vars(ap.parse_args())

print("[INFO] downloading IMDB...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

start_time = time.time()

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print("[INFO] compiling model...")
model = IMDB.build(top_words, embedding_vector_length, max_review_length)

if args["batch_size"] > 0:
    batch_size = args["batch_size"]

if args["epochs"] > 0:
    epochs = args["epochs"]

if args["gpu"] > 0:
    gpus = ["gpu(" + str(index) + ")" for index in range(args["gpu"])]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],
                  context=gpus)
else:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("[INFO] training...")
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                    verbose=1, validation_data=(X_test, y_test))

accuracy = history.history['acc'][-1]
loss = history.history['loss'][-1]

print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

end_time = time.time()
time_elapsed = end_time - start_time
print("[INFO] time: {:.2f}".format(time_elapsed))

file_results = open("./output/results.txt", "a")
file_results.write(
    str(batch_size) + "," + str(epochs) + "," + str(loss) + "," + str(accuracy) + "," + str(time_elapsed) + "\n")
file_results.close()
