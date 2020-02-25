import pandas as pd
import os
import re
from ast import literal_eval
import gensim
import numpy as np
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import Dataset, DataLoader

## Fetching and cleaning data
path = '../data_file.csv'
df = pd.read_csv(path)
data = df.Review.map(lambda x : literal_eval(x)).tolist()
data = [items for items in data if not items == []]
data = sum(data, [])
to_be_removed = '!"#$%&()*+-:;<=>?@[\]^_`{|}~'
to_be_kept = '.,\'/'

def clean_sentence(review):
    review = review.strip(' ').replace('\x92', '\'').replace('\n', '.').replace('\r', ' ').lower()
    review = re.sub(r'[.]+', '.', review)
    for items in to_be_removed :
        review = review.replace(items, '')
    revs = review.split('.')
    li = []
    for reviews in revs :
        for items in to_be_kept :
            reviews = reviews.replace(items, ' '+items+' ')
        li.append(reviews)
    return li

data = [clean_sentence(items) for items in data]
data = sum(data, [])
data = [items.split(' ') for items in data]
data = [[i for i in items if not i == ''] for items in data]
data = [items for items in data if len(items) > 1]
data = [items for items in data if len(items) < 20]
data = [items for items in data if not items == 'there are no comments available for this review'.split(' ')]


class Language :
    vocab = 0
    word2index = 0
    index2word = 0
    embeddings = 0

    def __init__(self, sentences, embed_size = None) :
        try :
            word2vec = gensim.models.KeyedVectors.load('word_embeddings_{}d.model'.format(embed_size))
            print('Using Pretrained Embeddings')
        except:
            print('Creating and using Pretrained Embeddings')
            word2vec = gensim.models.Word2Vec(sentences = sentences,
                                size = embed_size, min_count = 2, window = 10, iter = 20)
            word2vec.save('word_embeddings_{}d.model'.format(embed_size))
        Language.vocab = list(word2vec.wv.vocab.keys())
        Language.vocab.extend(['<UNK>', '<SOS>', '<EOS>'])
        # for items in range(word2vec.wv.vectors.shape[0]):
        #     word2vec.wv.vectors[items] = normalize(word2vec.wv.vectors[items, np.newaxis])
        Language.embeddings = word2vec.wv.vectors
        Language.word2index = self.__make_dict(Language.vocab, 'w2i')
        Language.index2word = self.__make_dict(Language.vocab, 'i2w')

    def __make_dict(self, vocab, key) :
        if key == 'w2i' :
            return {items : i for i, items in enumerate(vocab)}
        elif key == 'i2w' :        
            return {i : items for i, items in enumerate(vocab)}

class Data(Dataset) :
    language = 0
    def __init__(self, sentences, train = False, embed_size = None) :
        if train :
            Data.language = Language(data, embed_size)
        self.data = sentences

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        sentence = self.data[idx]
        inputs = [Data.language.word2index[items]
                  if items in Data.language.word2index else Data.language.word2index['<UNK>']
                  for items in sentence]
        inputs.append(Data.language.word2index['<EOS>'])
        # while len(inputs) < 20 :
        #     inputs.append(Data.language.word2index['<PAD>'])
        return (torch.tensor(inputs), torch.tensor(inputs))
    

        
def dataloader(split = 0.7, valid = True, embed_size = 600) :
    assert isinstance(data, list), 'Data Should be a list'
    test_span = valid_span = train_span = 0
    train_length = int(len(data)*split)
    train_dataset = Data(data[:train_length], True, embed_size)
    train_loader = DataLoader(train_dataset, shuffle = True)
    print('Length of training_data :- ', len(train_loader))

    if valid :
        valid_length = int(len(data)*(1-split)/2)
        valid_span = data[train_length:train_length+valid_length]
        test_span = data[train_length+valid_length:]

        valid_dataset = Data(valid_span)
        valid_loader = DataLoader(valid_dataset, shuffle = True)
        print('Length of validatin_data :- ', len(valid_loader))

        test_dataset = Data(test_span)
        test_loader = DataLoader(test_dataset, shuffle = False)
        print('Length of testing_data :- ', len(test_loader))

        return train_loader, valid_loader, test_loader

    else :
        test_span = data[train_length:]
        test_dataset = Data(False, test_span)
        test_loader = DataLoader(test_dataset, shuffle=False)
        print('Length of testing_data :- ', len(test_loader))

        return train_loader, None, test_loader