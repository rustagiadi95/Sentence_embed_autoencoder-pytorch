import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Data

class Encoder1(nn.Module):
    def __init__(self, embed_dim, vocab_len, output_len, lstm_layers) :
        super(Encoder1, self).__init__()
        self.output_len = output_len
        self.embedding = nn.Embedding(vocab_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, output_len, lstm_layers, bidirectional=True)

    def forward(self, x) :
        embeddings = self.embedding(x)
        embeddings = embeddings.transpose(0, 1)
        out, hidden = self.lstm(embeddings)

        return out, hidden

class Decoder1(nn.Module):
    def __init__(self, embed_dim, vocab_len, output_len, lstm_layers) :
        super(Decoder1, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, output_len, lstm_layers, bidirectional=True)
        self.final_layer = nn.Linear(2*output_len, vocab_len)

    def forward(self, x, hidden) :
        embeddings = self.embedding(x)
        embeddings = embeddings.unsqueeze(0).unsqueeze(0) if len(embeddings.size()) == 1 else embeddings
        out, hidden = self.lstm(embeddings, hidden)
        output = F.softmax(self.final_layer(out.squeeze(0)), dim = 1)
        return output, hidden

def load_pretrained(encoder, decoder, embed_dim):
    print('Loading Pretrained Embeddings')
    temp = torch.zeros(len(Data.language.embeddings)+3, embed_dim)
    for i in range(len(Data.language.embeddings)) :
        temp[i] = torch.tensor(Data.language.embeddings[i])
    encoder.embedding.weight.data = torch.Tensor(temp).to(torch.device('cuda'))
    decoder.embedding.weight.data = torch.Tensor(temp).to(torch.device('cuda'))