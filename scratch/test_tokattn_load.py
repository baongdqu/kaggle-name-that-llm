import torch
from torch import nn

class TokAttn(nn.Module):
    def __init__(self, H, num_labels=18):
        super().__init__()
        self.drop = nn.Dropout(0.2)
        self.proj = nn.Linear(H, 64)
        self.score = nn.Linear(64, 1)
        self.out = nn.Linear(128, num_labels)

    def forward(self, x, m):   # x (B, T, H) standardized, m (B, T) bool
        h = torch.tanh(self.proj(self.drop(x)))
        s = self.score(h).squeeze(-1).masked_fill(~m, -1e4)
        a = torch.softmax(s, -1).unsqueeze(-1)
        mf = m.unsqueeze(-1).float()
        mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([mean, (a * h).sum(1)], -1)))

pt = torch.load('scratch/out_tokattn2/tokattn_debl23_final.pt', map_location='cpu')
net = TokAttn(pt['hidden'], len(pt['classes']))
net.load_state_dict(pt['state_dict'])
net.eval()
print("TokAttn DeBERTa L23 net loaded successfully! Params:", sum(p.numel() for p in net.parameters()))

