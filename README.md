# Sentence_embed_autoencoder-pytorch
Creating the sentence embedding using the auto-encoders in pytorch

The dataset used here are hotel reviews obtained online.
The autoencoder here as **bi-LSTMs** as the encoder and decoder with no dropout.
The word embeddings are generated using [gensim's word2vec](https://radimrehurek.com/gensim/models/word2vec.html).

The accuracy metric is BLEU score using the smoothing function from [nltk](https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.bleu_score).

---

### Package Requirements
* Python 3.6.4
* pytorch 1.2.0+cu92
* matplotlib 3.1.2
* pandas 0.22.0
* re 2.2.1
* gensim 3.8.1
* numpy 1.17.0
* sklearn 0.19.1
* nltk 3.2.5

---

### How To Run
`python run.py`