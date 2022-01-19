import numpy as np
import pathlib
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from chowder import CHOWDER, Dataset
from submission_tools import get_train_set

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')

X_train, y_train = get_train_set(data_dir)
X_train = torch.from_numpy(X_train.astype(np.float32).reshape((-1, 2048, 1000)))

n_runs = 2
learning_rate = 1e-3
n_epochs = 30
l2_regularization = .5

training_set = Dataset(X_train, y_train)
generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

loss_function = nn.BCELoss(reduction='sum')

preds = []
for _ in range(n_runs):
    model = CHOWDER(dropout_2=.23).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        for X_batch, y_batch in generator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.float)
            
            y_pred = model(X_batch)
            
            conv_params = torch.cat([x.view(-1) for x in model.feature_embedding.parameters()])
            l2_conv = torch.sum(conv_params ** 2)
            loss = loss_function(y_pred, y_batch) + l2_regularization * l2_conv
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

    y_pred = model.predict_proba(X_train.to(device))
    preds.append(y_pred)

mean_pred = np.stack(preds).mean(axis=0)
print(roc_auc_score(y_train, mean_pred))
