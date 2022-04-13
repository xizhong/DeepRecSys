# @Time  : 2022/3/30 20:56
# @Author: xizhong
# @Desc  : https://blog.csdn.net/BIT_666/article/details/116459627

"""
There are 2 input for QA
question: convert text to Embedding and LSTM
answer: convert text to Embedding and LSTM
the two to concatenate and pass the to dense
dense output to softmax to classification
"""

import tensorflow as tf
import numpy as np
# tf.compat.v1.disable_eager_execution() #不能调用RNN

## models build
text_vocabulary_size = 10
question_vocabulary_size = 10
answer_vocabulary_size = 5

text_input = tf.keras.layers.Input(shape=(None, ), dtype='int32', name='text')
embed_text = tf.keras.layers.Embedding(input_dim=text_vocabulary_size, output_dim=64)(text_input)
encode_text = tf.keras.layers.LSTM(32)(embed_text)
question_input = tf.keras.layers.Input(shape=(None, ), dtype='int32', name='question')
embed_question = tf.keras.layers.Embedding(input_dim=question_vocabulary_size, output_dim=32)(question_input)
encode_question = tf.keras.layers.LSTM(16)(embed_question)

concatenated = tf.keras.layers.concatenate([encode_text, encode_question], axis=-1)

answer = tf.keras.layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
model = tf.keras.Model(inputs=[text_input, question_input], outputs=answer)
model.summary()

if __name__ == '__main__':
    num_samples = 1000
    max_length = 100

    text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
    answers = np.random.randint(answer_vocabulary_size, size=(num_samples))

    answers = tf.keras.utils.to_categorical(y=answers, num_classes=answer_vocabulary_size)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

# Multi output https://blog.csdn.net/BIT_666/article/details/116480208

# Loss https://blog.csdn.net/BIT_666/article/details/108796486