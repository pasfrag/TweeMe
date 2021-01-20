import tensorflow as tf
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=20000)

def dataset_preparation():

    data_dir = pathlib.Path('D:\Dataset\Final2')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=10,
        image_size=(240, 240),
        label_mode='binary',
        batch_size=16)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=10,
        image_size=(240, 240),
        label_mode='binary',
        batch_size=16)
    return train_ds, val_ds


def split_and_preprocess(df):

    tokenizer.fit_on_texts(df['preprocessed_text'].values)
    word_index = tokenizer.word_index
    print('Dataset contains %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['preprocessed_text'].values)
    X = pad_sequences(X, maxlen=50)
    Y = df['label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    X_train = np.asarray(X_train).astype(np.int)
    X_test = np.asarray(X_test).astype(np.int)
    Y_train = np.asarray(Y_train).astype(np.int)
    Y_test = np.asarray(Y_test).astype(np.int)
    return X_train, X_test, Y_train, Y_test
    # return X, Y
