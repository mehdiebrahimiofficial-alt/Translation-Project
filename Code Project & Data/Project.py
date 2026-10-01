import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

## Read and Information of Data

Data=pd.read_csv('eng-french.csv')

print(Data.info())
print(Data.head())

## X , Y , Split data
from sklearn.model_selection import train_test_split

English=Data['English words/sentences'].astype('str')
French=Data['French words/sentences'].astype('str')

French=French.apply(
    lambda x : '<start> ' + x + ' <end>'
)

eng_train,eng_temp,fr_train,fr_temp=train_test_split(English,French,test_size=0.2,random_state=42)
eng_val,eng_test,fr_val,fr_test=train_test_split(eng_temp,fr_temp,test_size=0.5,random_state=42)

## Tokenizer
import tensorflow as tf
import keras
from keras import layers

english_tokenizer=layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=20,
    output_mode='int'
)

french_tokenizer=layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=20,
    output_mode='int',
    standardize=None,
    split='whitespace'
)

english_tokenizer.adapt(eng_train)
french_tokenizer.adapt(fr_train)

eng_train_tokens=english_tokenizer(eng_train)
eng_test_tokens=english_tokenizer(eng_test)
eng_val_tokens=english_tokenizer(eng_val)

fr_train_tokens=french_tokenizer(fr_train)
fr_test_tokens=french_tokenizer(fr_test)
fr_val_tokens=french_tokenizer(fr_val)

## Positional Encoder Layer

class PositionalEncoder(layers.Layer):
    def __init__(self,seq_length,d_model):
        super().__init__()
        self.supports_masking=True
        position=np.arange(seq_length)[:,np.newaxis]
        dimension=np.arange(d_model)[np.newaxis,:]
        angel_rates=1 / np.power(
            10000,(2*(dimension//2)/d_model)
            )
        
        angle_rads= angel_rates * position
        angle_rads[:,0::2]=np.sin(angle_rads[:,0::2])
        angle_rads[:,1::2]=np.cos(angle_rads[:,1::2])

        self.pos_encoding=tf.constant(
            angle_rads,
            dtype=tf.float32
        )

    def call(self,inputs):
        return inputs + self.pos_encoding

## Encoder Block 
def Transformer_Encoder(
        inputs,
        d_model=64,
        num_heads=2,
        ffn_dim=128,
        dropout=0.1,
):
    attention_output=layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model//num_heads
    )(inputs,inputs)

    x = layers.add([inputs , attention_output])

    x=layers.LayerNormalization()(x)

    x=layers.Dropout(dropout)(x)

    ffn=layers.Dense(ffn_dim,activation='relu')(x)
    ffn=layers.Dense(d_model)(ffn)

    x = layers.add([x , ffn])

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(dropout)(x)

    return x

## Decoder Block

def Transformer_Decoder(
        decoder_inputs,
        encoder_outputs,
        num_heads=2,
        d_model=64,
        ffn_dim=128,
        dropout=0.1
):
    self_attention=layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model//num_heads
    )(decoder_inputs,decoder_inputs,use_causal_mask=True)

    x = layers.add([ decoder_inputs , self_attention])

    x = layers.LayerNormalization()(x)

    x = layers.Dropout(dropout)(x)

    cross_attention=layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model//num_heads
    )(x,encoder_outputs)

    x2 = layers.add([x , cross_attention])

    x2 = layers.LayerNormalization()(x2)

    x2 = layers.Dropout(dropout)(x2)

    ffn=layers.Dense(ffn_dim,activation='relu')(x2)
    ffn=layers.Dense(d_model)(ffn)

    output= layers.add([x2 , ffn])

    output= layers.LayerNormalization()(output)

    output = layers.Dropout(dropout)(output)

    return output

## Encoder
encoder_inputs=keras.Input(
    shape=(20,),
    dtype='int32'
)

encoder_embedding=layers.Embedding(
    mask_zero=True,
    input_dim=5000,
    output_dim=64
)(encoder_inputs)

encoder_embedding=PositionalEncoder(
    seq_length=20,
    d_model=64
)(encoder_embedding)

encoder_output=Transformer_Encoder(
    encoder_embedding,
    num_heads=2,
    d_model=64,
    ffn_dim=128,
    dropout=0.1
)

## Decoder
decoder_inputs=keras.Input(
    shape=(19,),
    dtype='int32'
)

decoder_embedding=layers.Embedding(
    mask_zero=True,
    input_dim=5000,
    output_dim=64
)(decoder_inputs)

decoder_embedding=PositionalEncoder(
    seq_length=19,
    d_model=64
)(decoder_embedding)

decoder_output=Transformer_Decoder(
    decoder_embedding,
    encoder_output,
    d_model=64,
    num_heads=2,
    ffn_dim=128,
    dropout=0.1
)

## OUTPUT layer

output=layers.Dense(5000,activation='softmax')(decoder_output)

## Train/Compile/Evaluate Model

Model=keras.Model([encoder_inputs,decoder_inputs],output)

Model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(Model.summary())

decoder_train_input=fr_train_tokens[:,:-1]
decoder_train_target=fr_train_tokens[:,1:]

decoder_val_input=fr_val_tokens[:,:-1]
decoder_val_target=fr_val_tokens[:,1:]

Training = Model.fit([eng_train_tokens,decoder_train_input],decoder_train_target,
validation_data=([eng_val_tokens,decoder_val_input],decoder_val_target),
epochs=10,batch_size=64)

decoder_test_input=fr_test_tokens[:,:-1]
decoder_test_target=fr_test_tokens[:,1:]

loss,accuracy=Model.evaluate([eng_test_tokens,decoder_test_input],decoder_test_target)

print('LOSS:',loss)
print('ACCURACY:',accuracy)

## Plots & Visualization
plt.figure(figsize=(12,8))
plt.plot(Training.history['loss'],label='Train loss')
plt.plot(Training.history['val_loss'],label='Validation loss')
plt.legend()
plt.title('Train & Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('Train & Validation loss.png')

plt.show()

plt.figure(figsize=(12,8))
plt.plot(Training.history['accuracy'],label='Train accuracy')
plt.plot(Training.history['val_accuracy'],label='Validation accuracy')
plt.legend()
plt.title('Train & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.savefig('Train & Validation Accuracy.png')

plt.show()

## Save Model & Informations

Model.save('Translate_transformer.keras')
print('Model is Saved!')

english_vocab=(english_tokenizer.get_vocabulary())

with open(
    'english_vocab.txt','w',encoding='utf-8'
) as file:

    for word in english_vocab:

        file.write(word + '\n')


french_vocab=(french_tokenizer.get_vocabulary())

with open(
    'french_vocab.txt','w',encoding='utf-8'
) as file:

    for word in french_vocab:

        file.write(word + '\n')

print('Vocabulary Saved Succesfully!')



