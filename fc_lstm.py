"""
Unofficial implementation of paper
'Generating Sequences With Recurrent Neural Networks'
https://arxiv.org/abs/1308.0850
"""
import torch
import torch.nn as nn


class FC_LSTM(nn.Module):
    """
    - Params: input_size, hidden_size
    - Input: x(b, t, f)
    - Output: hiddens(length: t), memorys(length: t)
    """

    def __init__(self, input_size, hidden_size):
        super(FC_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc_x = nn.Linear(input_size, 4 * hidden_size)
        self.fc_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_ci = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_cf = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_co = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, x, init_hidden, init_memory):
        batch_size, seq_len, input_size = x.size()
        hiddens = []
        memorys = []
        hiddens.append(init_hidden)
        memorys.append(init_memory)

        for t in range(seq_len):
            x_t = x[:, t, :]

            x_t_w = self.fc_x(x_t)
            h_t_w = self.fc_h(hiddens[t])
            y = x_t_w + h_t_w
            i, f, o, g = torch.split(y, self.hidden_size, dim=1)

            Ci = torch.mul(self.W_ci, memorys[t])
            Cf = torch.mul(self.W_cf, memorys[t])

            i = torch.sigmoid(i + Ci)
            f = torch.sigmoid(f + Cf)
            g = torch.tanh(g)

            memory = torch.mul(f, memorys[t]) + torch.mul(i, g)
            Co = torch.mul(self.W_co, memory)
            o = torch.sigmoid(o + Co)
            hidden = torch.mul(o, torch.tanh(memory))

            hiddens.append(hidden)
            memorys.append(memory)

        hiddens = torch.stack(hiddens, dim=0)
        memorys = torch.stack(memorys, dim=0)

        return hiddens[1:], memorys[1:]


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, time sequence length, features)
    inputs = torch.randn(32, 12, 207).to(device)

    print('FC-LSTM Layer usage:')
    b, t, f = inputs.shape
    hidden_size = 64
    init_hidden = torch.zeros(b, hidden_size).to(device)
    init_memory = torch.zeros(b, hidden_size).to(device)
    fc_lstm = FC_LSTM(f, hidden_size)
    fc_lstm.to(device)
    x_fc_lstm, memorys = fc_lstm(inputs, init_hidden, init_memory)
    print('Length of hidden states', len(x_fc_lstm))
    print('Length of memorys', len(memorys))
    print('Shape of output:', x_fc_lstm.shape)
    print('Shape of all memorys:', memorys.shape)
