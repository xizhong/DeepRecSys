# @Time  : 2022/3/30 22:02
# @Author: xizhong
# @Desc  : https://blog.csdn.net/BIT_666/article/details/105812443

import tensorflow as tf

texts, labels = [
    '77,Nico Icon (1995),Documentary',
    '78,Crossing Guard, The (1995),Action|Crime|Drama|Thriller',
    '79,Juror, The (1996), Drama|Thriller',
    '80,"White Balloon, The (Badkonake sefid) (1995)",Children|Drama',
    '81,Things to Do in Denver When You are Dead (1995),Crime|Drama|Romance'
], [1, 0, 1, 0, 1, 0]

max_words = 10  # high frequency words
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts=texts)
# to sequence [[6, 7, 8, 1, 9], [3, 1, 4, 2, 5], [3, 2, 5], [3, 1, 2], [1,
# 4, 2]]
sequences = tokenizer.texts_to_sequences(texts)
# print(sequence)

# word_index = tokenizer.word_index  # all unique word to index {'the', 3}
# print(word_index)

# padding sequence


def padding_sequences(
        sequences,
        max_len=None,
        dtype='int32',
        padding='post',
        truncating='pre',
        value=0):
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_len,
        dtype=dtype,
        padding=padding,
        truncating=truncating,
        value=value)
    return sequences


text_sequences = padding_sequences(sequences, max_len=10)
labels = tf.keras.utils.to_categorical(labels)
# print(labels)

# generate_model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10, 4, input_length=10))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()
model.fit(text_sequences, labels, epochs=3, batch_size=2, validation_split=0.2)
