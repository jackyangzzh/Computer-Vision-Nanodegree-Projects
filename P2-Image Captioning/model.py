import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 0, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = self.embedding(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), captions), 1)
        outputs,(h_n, c_n) = self.lstm(inputs)
        outputs = self.linear(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted = []
        for i in range(max_len):
            # LSTM 
            hidden, states = self.lstm(inputs, states)
            # Linear
            output = self.linear(hidden.squeeze(1))
            # Max
            word = output.max(1)[1]
            predicted.append(word.item())
            inputs = self.embedding(word)
            inputs = inputs.unsqueeze(1)
        return predicted