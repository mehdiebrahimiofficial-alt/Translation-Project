# English → French Translation with a Transformer

A small encoder–decoder Transformer built with Keras/TensorFlow that translates short English sentences into French.

## Project Structure

```
.
├── Project.py                   # Data prep, model, training, evaluation, saving
├── Translate.py                 # Load the saved model and translate sentences
├── eng-french.csv               # Dataset (English / French sentence pairs)
├── Translate_transformer.keras  # Saved model (created by Project.py)
├── english_vocab.txt            # English vocabulary (created by Project.py)
├── french_vocab.txt             # French vocabulary (created by Project.py)
└── *.png                        # Loss / accuracy plots (created by Project.py)
```

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn, Matplotlib

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Dataset

`eng-french.csv` with two columns:

- `English words/sentences`
- `French words/sentences`

French sentences are wrapped with `<start>` and `<end>` tokens. The data is split 80% train / 10% validation / 10% test (`random_state=42`).

## Model Architecture

| Component | Setting |
|---|---|
| Vocabulary size | 5000 (each language) |
| Sequence length | 20 (encoder) / 19 (decoder) |
| Embedding size (`d_model`) | 64 |
| Attention heads | 2 |
| Feed-forward size | 128 |
| Dropout | 0.1 |
| Encoder / Decoder blocks | 1 / 1 |
| Positional encoding | Sinusoidal |
| Optimizer / Loss | Adam / Sparse categorical crossentropy |
| Training | 10 epochs, batch size 64 |

The decoder uses causal self-attention plus cross-attention over the encoder output, followed by a softmax layer over the French vocabulary.

## Usage

### 1. Train

```bash
python Project.py
```

This trains the model, evaluates it on the test set, saves the plots, the model, and both vocabulary files.

### 2. Translate

```bash
python Translate.py
```

Or from code:

```python
print(translate("i love you"))
```

## How Inference Works

Translation uses **greedy decoding**:

1. The English sentence is tokenized and passed to the encoder.
2. The decoder input is a zero-filled array of length **19** with `<start>` at position 0.
3. At each step `i`, the model predicts the next token from **position `i`** of the output.
4. The predicted token is written to position `i + 1` of the decoder input.
5. The loop stops at `<end>` or padding (id 0), or after 18 steps.

> **Important:** the model was built with a fixed decoder input shape of `(19,)`, so the decoder input must always have length 19 at inference. Feeding a growing sequence and reading the last position (`[-1]`) produces empty or meaningless output.

## Notes and Possible Improvements

- Reported accuracy includes padding positions, so it overestimates real quality. Use a masked loss/accuracy that ignores PAD tokens.
- Multiply embeddings by `sqrt(d_model)` before adding positional encoding.
- Lowercase the text and separate punctuation from words to reduce `[UNK]` tokens.
- Use more encoder/decoder layers, more epochs, and longer sequences for better translations.
- Beam search can improve results over greedy decoding.

