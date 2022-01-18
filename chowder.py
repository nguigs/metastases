import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class CHOWDER(nn.Module):
    def __init__(self, retain=2):
        super(CHOWDER, self).__init__()
        self.feature_embedding = nn.Parameter(torch.randn(2048))
        self.retain = retain
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * retain, 200), nn.Sigmoid(),
            nn.Linear(200, 100), nn.Sigmoid(),
            nn.Linear(100, 1), nn.Softmax(dim=1))

    def forward(self, x):
        embedded = torch.matmul(x, self.feature_embedding)
        embedded_sorted, _ = torch.sort(embedded)
        min = embedded_sorted[..., :self.retain]
        max = embedded_sorted[..., -self.retain:]
        max_min = torch.cat([min, max], dim=-1)
        return self.fully_connected(max_min)
