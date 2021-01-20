import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Dropout, Conv1D, Embedding, MaxPooling1D, LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.resnet import ResNet50

from utilities import tokenizer
from visualizations import plot_history
import numpy as np


# def vgg_model(x_train, y_train, x_test, y_test, trainable=True):
def vgg_model(train_ds, val_ds, trainable=True):
    base_model = VGG19(input_shape=(240, 240, 3), include_top=False, weights='imagenet')

    if not trainable:
        for layer in base_model.layers:
            layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(base_model.input, x)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    vgg_history = model.fit(train_ds, validation_data=val_ds, steps_per_epoch=16, epochs=10)
    try:
        plot_history(vgg_history, "vgg_history")
    except:
        pass
    model.save('saved-models/VGG19c_b16')


def inception_model(train_ds, val_ds, trainable=True):
    base_model = InceptionV3(input_shape=(240, 240, 3), include_top=False, weights='imagenet')

    if not trainable:
        for layer in base_model.layers:
            layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    inception_history = model.fit(train_ds, validation_data=val_ds, steps_per_epoch=16, epochs=10)
    try:
        plot_history(inception_history, "inception_history")
    except:
        pass
    model.save('saved-models/InceptionV3c_b16')


def resnet_model(train_ds, val_ds, trainable=True):
    base_model = ResNet50(input_shape=(240, 240, 3), include_top=False, weights='imagenet')

    if not trainable:
        for layer in base_model.layers:
            layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    inception_history = model.fit(train_ds, validation_data=val_ds, steps_per_epoch=16, epochs=10)
    try:
        plot_history(inception_history, "resnet_history")
    except:
        pass
    model.save('saved-models/ResNet50c_b16')


def rnn_model(X_train, X_test, y_train, y_test):

    # Read the pretrained embeddings from glove
    embeddings_idx = dict()
    file = open(
        'glove.twitter.27B.200d.txt',
        encoding='utf-8')
    for line in file:
        vals = line.split()
        word = vals[0]
        numbers = np.asarray(vals[1:], dtype='float32')
        embeddings_idx[word] = numbers
    file.close()

    # Create matrix for the pretrained embeddings
    embeddings = np.zeros((20000, 200))
    for token, idx in tokenizer.word_index.items():
        if idx > 20000 - 1:
            break
        else:
            vector = embeddings_idx.get(token)
            if vector is not None:
                embeddings[idx] = vector

    glove = Sequential()
    glove.add(Embedding(20000, 200, input_length=50, weights=[embeddings], trainable=False))
    glove.add(Dropout(0.2))
    glove.add(Conv1D(64, 5, activation='relu'))
    glove.add(MaxPooling1D(pool_size=4))
    glove.add(LSTM(100, return_sequences=True))
    glove.add(Dropout(0.2))
    glove.add(LSTM(100))
    glove.add(Dense(1, activation='sigmoid'))
    glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # Train the model and make predictions
    glove_history = glove.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
    glove.fit()

    try:
        plot_history(glove_history, "rnn_history")
    except:
        pass
    glove.save('saved-models/RNN_10')
