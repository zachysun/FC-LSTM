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
    - Output: hiddens(length: 1+t), memorys(length: 1+t)
    """

    def __init__(self, input_size, hidden_size):
        super(FC_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc_x = nn.Linear(input_size, 4 * hidden_size)
        self.fc_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_ci = nn.Linear(hidden_size, hidden_size)
        self.W_cf = nn.Linear(hidden_size, hidden_size)
        self.W_co = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        hiddens = []
        memorys = []
        init_hidden = torch.zeros(batch_size, self.hidden_size).cuda()
        init_memory = torch.zeros(batch_size, self.hidden_size).cuda()
        hiddens.append(init_hidden)
        memorys.append(init_memory)

        for t in range(seq_len):
            x_t = x[:, t, :]

            x_t_w = self.fc_x(x_t)
            h_t_w = self.fc_h(hiddens[t])
            y = x_t_w + h_t_w
            i, f, o, g = torch.split(y, self.hidden_size, dim=1)

            Ci = self.W_ci(memorys[t])
            Cf = self.W_cf(memorys[t])

            i = torch.sigmoid(i + Ci)
            f = torch.sigmoid(f + Cf)
            g = torch.tanh(g)

            memory = f * memorys[t] + i * g
            Co = self.W_co(memory)
            o = torch.sigmoid(o + Co)
            hidden = o * torch.tanh(memory)

            hiddens.append(hidden)
            memorys.append(memory)

        return hiddens, memorys


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, time sequence length, features)
    inputs = torch.randn(32, 12, 207).to(device)

    print('FC-LSTM Layer usage:')
    b, t, f = inputs.shape
    hidden_size = 64
    fc_lstm = FC_LSTM(f, hidden_size)
    fc_lstm.to(device)
    x_fc_lstm, memorys = fc_lstm(inputs)
    print('Length of hidden states', len(x_fc_lstm))
    print('Length of memorys', len(memorys))
    print('Output shape of FC-LSTM Layer(final hidden states):', x_fc_lstm[t-1].shape)
