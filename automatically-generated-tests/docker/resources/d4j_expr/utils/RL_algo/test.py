import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import torch.nn as nn

a = torch.ones(25, 300)
b = torch.ones(22, 300)
c = torch.ones(15, 300)

# print([a,b,c])
# print(torch.stack([a,b]))
lstm = nn.LSTM(input_size=300, hidden_size = 10, num_layers=1)
seq_len = [s.size(0) for s in [a,b,c]]
d = pad_sequence([a, b, c])
# print(d.size())
# # assert d[:,0,:].all() == a.all()
e = pack_padded_sequence(d, seq_len, enforce_sorted=False)
f = pack_sequence([a,b,c], enforce_sorted=False)
print(e)
print(f)
# # print(e)
out, (h_n, c_n) = lstm(f)
# print(h_n.squeeze(0).size())
print(h_n.squeeze(0).size())
# # print(out.size())

# real_out, lens = pad_packed_sequence(out)
# print(real_out.size())
# assert h_n[0, 2, :].all() == real_out[10, 2, :].all()
# print(real_out.size())
# print(h_n)
# seq_len = [s.size(0) for s in [a,b,c]]
# print(pack_padded_sequence(d, seq_len, batch_first=True, enforce_sorted=False))
