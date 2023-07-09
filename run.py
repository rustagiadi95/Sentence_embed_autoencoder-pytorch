import torch
import torch.nn as nn
import torch.optim as optim

from models import Encoder1, Decoder1, load_pretrained
from data import dataloader, Data
import os
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
from tqdm import tqdm




var = {
    'valid' : True,
    'embed_dim' : 600,
    'train_test_split' : 0.75,
    'lstm_layers' : 2,
    'latent_dim' : 600
}

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('--------------- Creating Data ---------------')

train_loader, valid_loader, test_loader = dataloader(var['train_test_split'], 
                                                      var['valid'],
                                                      var['embed_dim'])
print('--------------- Done !! ---------------')

print('--------------- Creating Models ---------------')
encoder = Encoder1(var['embed_dim'], len(Data.language.vocab), var['latent_dim'], var['lstm_layers'])
encoder = encoder.to(device)
decoder = Decoder1(var['latent_dim'], len(Data.language.vocab), var['embed_dim'], var['lstm_layers'])
decoder = decoder.to(device)
current_epoch = 0
load_pretrained(encoder, decoder, var['embed_dim'])
if not len(os.listdir('Saved_Model/')) == 0 :
    li = [int(items.split('_')[1].strip('.pt')) for items in os.listdir('Saved_Model/')]
    li = list(set(li))
    li.sort()
    current_epoch = li[-1]
    print('Loading model for epoch {}'.format(current_epoch))
    encoder.load_state_dict(torch.load('Saved_Model/encoder1_{}.pt'.format(current_epoch)))
    decoder.load_state_dict(torch.load('Saved_Model/decoder1_{}.pt'.format(current_epoch)))

print(encoder)
print(decoder)
print('--------------- Done !! ---------------')
lr = 0.0001
criterion = nn.CrossEntropyLoss().to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr = lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = lr)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr
    

log = open('logs.txt', 'a')
print('--------------- Training ---------------')

start_epoch = current_epoch
end_epoch = 100
max_score = 0
testing_output = []
# teacher_forcing_ratio = 0.5
smoothie = SmoothingFunction().method4

for epoch in tqdm(range(start_epoch, end_epoch)):
    
    total_loss = 0
    total_bleu = 0
    encoder.train()
    decoder.train()
    new_lr = adjust_learning_rate(encoder_optimizer, epoch)
    adjust_learning_rate(decoder_optimizer, epoch)

    log.write('The Learning Rate is : {}\n'.format(new_lr))
    training_output = []
    for idx, (inputs, labels) in enumerate(train_loader) :
        encoder.zero_grad()
        decoder.zero_grad()
        inputs = inputs.to(device)
        output, hidden = encoder(inputs)
        labels = labels.to(device)
        decoder_input = torch.tensor([[Data.language.word2index['<SOS>']]], device=device)
        decoder_hidden = hidden
        loss = 0
        output_sent = []
        label_sent = []

        # if idx % 2 == 0 :
        #     for items in range(len(labels[0]) - 1) :
        #         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        #         loss += criterion(decoder_output, labels[0][items].unsqueeze(0).to(device))
        #         topv, topi = decoder_output.topk(1)
        #         output_sent.append(Data.language.index2word[topi.item()])
        #         label_sent.append(Data.language.index2word[labels[0][items].item()])
        #         decoder_input = labels[0][items]

        # else :
        for items in range(len(labels[0])) :
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, labels[0][items].unsqueeze(0).to(device))
            topv, topi = decoder_output.topk(1)
            output_sent.append(Data.language.index2word[topi.item()])
            label_sent.append(Data.language.index2word[labels[0][items].item()])
            decoder_input = topi.squeeze().detach()
        try :
            training_output.append((' '.join(output_sent), ' '.join(label_sent)))
            total_bleu += bleu([label_sent], output_sent, smoothing_function=smoothie)
            total_loss += loss.item()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        except Exception as e:
            print(e)
            print(output_sent)

    try :
        train_log = open('training_outputs.txt', 'w', encoding = 'latin')
        train_log.write('Recording outputs for epoch {}\n'.format(epoch + 1))
        train_log.write('Training_loss {}\n'.format(total_loss/len(train_loader)))
        train_log.write('Training_Accuracy {}\n'.format(total_bleu/len(train_loader)))
        for i, (out, lab) in enumerate(training_output):
            train_log.write('------------------------------------------------\n')
            train_log.write('Index: {}\n'.format(i))
            train_log.write('Output: {}\n'.format(out))
            train_log.write('Label: {}\n'.format(lab))
        train_log.flush()
        train_log.close()
    except:
        train_log.write("error in printing")
        train_log.flush()
        train_log.close()

    string = 'Training Epoch: {}/{}, Training Loss: {}, Training Accuracy: {}\n' \
              .format(epoch + 1, end_epoch, total_loss/len(train_loader), total_bleu/len(train_loader))
    log.write(string)
    with torch.no_grad() :
        encoder.eval()
        decoder.eval()
        if valid_loader :
            try :
                validating_output = []
                total_loss = 0
                total_bleu = 0
                for idx, (inputs, labels) in enumerate(valid_loader) :
                    inputs = inputs.to(device)
                    output, hidden = encoder(inputs)
                    labels = labels.to(device)
                    decoder_input = torch.tensor([[Data.language.word2index['<SOS>']]], device=device)
                    # decoder_hidden = (output.unsqueeze(0).unsqueeze(0),torch.zeros(1, 1, var['embed_dim'], device = device))
                    decoder_hidden = hidden
                    loss = 0
                    output_sent = []
                    label_sent = []
                    for items in range(len(labels[0])) :
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                        loss += criterion(decoder_output, labels[0][items].unsqueeze(0).to(device))
                        topv, topi = decoder_output.topk(1)
                        output_sent.append(Data.language.index2word[topi.item()])
                        label_sent.append(Data.language.index2word[labels[0][items].item()])
                        decoder_input = topi.squeeze().detach()
                    validating_output.append((' '.join(output_sent), ' '.join(label_sent)))
                    total_loss += loss.item()
                    total_bleu += bleu([label_sent], output_sent, smoothing_function=smoothie)

                if total_bleu/len(valid_loader) > max_score :
                    valid_log = open('validating_outputs.txt', 'w', encoding = 'latin')
                    valid_log.write('Recording outputs for epoch {}\n'.format(epoch + 1))
                    valid_log.write('Validating_loss {}\n'.format(total_loss/len(valid_loader)))
                    valid_log.write('Validating_accuracy {}\n'.format(total_bleu/len(valid_loader)))
                    for i, (out, lab) in enumerate(validating_output):
                        valid_log.write('------------------------------------------------\n')
                        valid_log.write('Index: {}\n'.format(i))
                        valid_log.write('Output: {}\n'.format(out))
                        valid_log.write('Label: {}\n'.format(lab))
                    valid_log.flush()
                    valid_log.close()
                    max_score = total_bleu/len(valid_loader)
            except:
                pass
            
            string = 'Validation Epoch: {}/{}, Validation Loss: {}, Validation Accuracy: {}\n' \
                      .format(epoch + 1, end_epoch, total_loss/len(valid_loader), total_bleu/len(valid_loader))
            log.write(string)
            log.write('----------------------------------------------------------------------------------------------\n')
            log.flush()
    torch.save(encoder.state_dict(), 'Saved_Model/encoder1_{}.pt'.format(epoch + 1))
    torch.save(decoder.state_dict(), 'Saved_Model/decoder1_{}.pt'.format(epoch + 1))
log.close()

print('--------------- Training Completed ---------------')
print('--------------- Testing the models ---------------')
encoder.eval()
decoder.eval()
with torch.no_grad():
    total_loss = 0
    total_bleu = 0
    for idx, (inputs, labels) in enumerate(test_loader) :
        inputs = inputs.to(device)
        output, hidden = encoder(inputs)
        labels = labels.to(device)
        decoder_input = torch.tensor([[Data.language.word2index['<SOS>']]], device=device)
        # decoder_hidden = (output.unsqueeze(0).unsqueeze(0),torch.zeros(1, 1, var['embed_dim'], device = device))
        decoder_hidden = hidden
        loss = 0
        output_sent = []
        label_sent = []
        for items in range(len(labels[0])) :
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, labels[0][items].unsqueeze(0).to(device))
            topv, topi = decoder_output.topk(1)
            output_sent.append(Data.language.index2word[topi.item()])
            label_sent.append(Data.language.index2word[labels[0][items].item()])
            decoder_input = topi.squeeze().detach()
        testing_output.append((' '.join(output_sent), ' '.join(label_sent)))
        total_loss += loss.item()
        total_bleu += bleu([label_sent], output_sent, smoothing_function=smoothie)

    test_log = open('testing_outputs.txt', 'w', encoding = 'latin')
    # test_log.write('Recording outputs for epoch {}\n'.format(epoch))
    test_log.write('Testing_loss {}\n'.format(total_loss/len(test_loader)))
    test_log.write('Testing_accuracy {}\n'.format(total_bleu/len(test_loader)))
    for i, (out, lab) in enumerate(testing_output):
        test_log.write('------------------------------------------------\n')
        test_log.write('Index: {}\n'.format(i))
        test_log.write('Output: {}\n'.format(out))
        test_log.write('Label: {}\n'.format(lab))
    test_log.flush()
    test_log.close()