from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding


class IMDB:
    @staticmethod
    def build(top_words, embedding_vector_length, max_review_length):
        model = Sequential()
        model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        model.add(Conv1D(32, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))

        return model