import torch.nn as nn
import torch
from torch.autograd import Variable

#####################
# Build model
#####################

class LSTM_jordan(nn.Module):

    def __init__(self, input_dim,
                 hidden_dim,
                 batch_size,
                 jordan_params,
                 output_dim=1,
                 num_layers=2):
        super(LSTM_jordan, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.n_jordan_params = jordan_params

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim + self.n_jordan_params, self.hidden_dim, self.num_layers)
        # context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
        self.jordan = nn.Parameter(torch.zeros(1, self.n_jordan_params))

        # Define the output layer
        # self.register_parameter('jordan', self.jordan)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        p1 = torch.zeros_like(input) + self.jordan[:, 0]
        p2 = torch.zeros_like(input) + self.jordan[:, 1]
        extended_input = torch.cat((input, p1, p2), dim=2)
        lstm_out, self.hidden = self.lstm(extended_input)
        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred, self.jordan

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()


# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
#         print(lstm_out.shape)
#         print("in")
#         print(input.shape)
        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
        
        # TODO maybe split computation graph here
        
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred
    
    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.prod(torch.Tensor(list(p.size()))) for p in model_parameters]).item()
