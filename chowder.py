import torch
from torch import nn


class CHOWDER(nn.Module):
    def __init__(self, retain=2, dropout_1=.5, dropout_2=.275):
        super(CHOWDER, self).__init__()
        self.feature_embedding = nn.Conv1d(2048, 1, kernel_size=3, padding='same')
        self.retain = retain
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * retain, 200), nn.Sigmoid(),
            nn.Linear(200, 100), nn.Dropout(p=dropout_1), nn.Sigmoid(),
            nn.Linear(100, 1), nn.Dropout(p=dropout_2), nn.Sigmoid())

    def forward(self, x):
        embedded = self.feature_embedding(x)
        min_feature, _ = torch.topk(embedded, self.retain, largest=False)
        max_feature, _ = torch.topk(embedded, self.retain)
        max_min = torch.cat([min_feature, max_feature], dim=-1)
        return self.fully_connected(max_min).flatten()

    def predict_proba(self, X):
        return self.forward(X).detach().cpu().numpy()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
