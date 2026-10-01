import numpy as np
import keras
from keras import layers

class PositionalEncoder(layers.Layer):
    def __init__(self, seq_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        pos = np.arange(seq_length)[:, None]
        dim = np.arange(d_model)[None, :]
        angles = pos / np.power(10000, 2 * (dim // 2) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = angles.astype('float32')

    def call(self, inputs):
        return inputs + self.pos_encoding


model = keras.models.load_model(
    'Translate_transformer.keras',
    custom_objects={'PositionalEncoder': PositionalEncoder}
)


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

english_vocab = load_vocab('english_vocab.txt')
french_vocab = load_vocab('french_vocab.txt')


english_tokenizer = layers.TextVectorization(
    max_tokens=5000, output_sequence_length=20, output_mode='int'
)
english_tokenizer.set_vocabulary(english_vocab)

START = french_vocab.index('<start>')
END = french_vocab.index('<end>')


def translate(sentence):
    enc = english_tokenizer([sentence])

    dec = np.zeros((1, 19), dtype='int32')
    dec[0, 0] = START

    words = []
    for step in range(18):
        pred = model.predict([enc, dec], verbose=0)
        next_id = int(np.argmax(pred[0, step]))

        if next_id in (0, END):
            break

        words.append(french_vocab[next_id])
        dec[0, step + 1] = next_id

    return ' '.join(words)

sentence= 'i love you'
Translasion= translate(sentence)
print('English:',sentence)
print('French:',Translasion)