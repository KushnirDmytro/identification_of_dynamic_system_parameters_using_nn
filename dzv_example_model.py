import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, cell, input_size, hidden_size, n_layers): # input_size  - vocabulary size
        super(EncoderRNN, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)  #emmbedding layer is trained also
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.recurrent(output, hidden)
        return output, hidden

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


class DecoderRNN(nn.Module):
    def __init__(self, cell, hidden_size, output_size, n_layers):
        super(DecoderRNN, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch_size=1):
        output = self.embedding(input).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


class AttnDecoderRNN(nn.Module):
    def __init__(self, cell, hidden_size, output_size, n_layers, max_length, dropout_emb):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_emb = dropout_emb
        # self.dropout_r = dropout_r
        self.max_length = max_length
        self.cell = cell
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_emb)
        if self.cell == 'GRU':
            self.recurrent = nn.GRU(hidden_size, hidden_size, n_layers,)
        elif self.cell == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, n_layers,)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        if self.cell == 'LSTM':
            cell_state = hidden[0]
        else:
            cell_state = hidden
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], cell_state[0]), 1)), dim=1) #instead of embedded - encoder outputs

        # print(encoder_outputs.shape)
        # print(embedded.shape)
        #
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((encoder_outputs, cell_state[0].repeat(self.max_length, 1, 1)), 1)), dim=1)


        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.view(batch_size, encoder_outputs.shape[0], self.hidden_size))

        output = torch.cat((embedded[0], attn_applied.view(1, batch_size, self.hidden_size)[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device, batch_size=1):
        if self.cell == 'LSTM':
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
                    )
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


class E2F(nn.Module):
    def __init__(self):
        super(E2F, self).__init__()

        self.drop = nn.Dropout(p=0.0)

        # self.fc0 = nn.Linear(512, 128)
        # self.fc1 = nn.Linear(128, 32)
        # self.fc2 = nn.Linear(32, 8)
        # self.fc3 = nn.Linear(8, 4)
        # self.fc4 = nn.Linear(4, 2)
        # self.fc5 = nn.Linear(2, 1)

        self.fc0 = nn.Linear(512, 64)
        self.fc1 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 2)
        self.fc5 = nn.Linear(2, 1)

        # self.fc0 = nn.Linear(512, 32)
        # self.fc3 = nn.Linear(32, 2)
        # self.fc5 = nn.Linear(2, 1)

    def forward(self, x):
        # h = F.selu(self.drop(self.fc0(x)))
        # h = F.selu(self.drop(self.fc1(h)))
        # h = F.selu(self.drop(self.fc2(h)))
        # h = F.selu(self.drop(self.fc3(h)))
        # h2 = F.selu(self.fc4(h))
        # h1 = self.fc5(h2)

        h = F.relu(self.drop(self.fc0(x)))
        h = F.relu(self.drop(self.fc1(h)))
        h2 = self.drop(self.fc3(h))
        h1 = self.fc5(h2)

        # h = F.selu(self.drop(self.fc0(x)))
        # h2 = F.selu(self.drop(self.fc3(h)))
        # # h2 = F.selu(self.fc3(h))
        # h1 = self.fc5(h2)
        return h1, h2