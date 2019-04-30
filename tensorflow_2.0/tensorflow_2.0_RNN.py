import tensorflow as tf
print(tf. __version__)

import numpy as np
import os
import time
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def plotgraphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel('string')
    plt.legend([string, 'val_'+string])
    plt.show()

# Setup input pipeline
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder
print('Vocabulary size: {}'.format(tokenizer.vocab_size))

sample_string = 'TensorFLow is cool.'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string. {}'.format(original_string))

assert original_string == sample_string

# 각 벡터에 해당하는 단어를 확인해보자
[print('{} -----> {}'.format(ts, tokenizer.decode([ts]))) for ts in tokenized_string]

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# 현재 아래 코드는 tensorflow 2.0-nightly 버전과 tfds-nightly에서만 실행이 가능하다.
# $ pip install tf-nightly-2.0-preview
# $ pip install tf-nightly-gpu-2.0-preview
# $ pip install tfds-nightly

train_dataset = train_dataset.shuffle(BATCH_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):
    tokenized_sample_pred_text = tokenizer.encode(sentence)

    if pad:
    tokenized_sample_pred_text = pad_to_size(tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return (predictions)

# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this moivie.')

predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this moivie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)