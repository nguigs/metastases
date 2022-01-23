import time
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class CHOWDER(nn.Module):
    def __init__(self, retain=2, dropout_0=.5, dropout_1=.5, dropout_2=.275, linear=True, relu=False):
        super(CHOWDER, self).__init__()
        if linear:
            self.feature_embedding = nn.Linear(2048, 1, bias=False)
        else:
            self.feature_embedding = nn.Conv1d(2048, 1, kernel_size=(3, ), padding='same')
        self.linear = linear
        self.retain = retain
        if relu:
            self.fully_connected = nn.Sequential(
                nn.Dropout(p=dropout_0), nn.Linear(2 * retain, 200), nn.ReLU(),
                nn.Dropout(p=dropout_1), nn.Linear(200, 100), nn.ReLU(),
                nn.Dropout(p=dropout_2), nn.Linear(100, 1))
        else:
            self.fully_connected = nn.Sequential(
                nn.Dropout(p=dropout_0), nn.Linear(2 * retain, 200), nn.Sigmoid(),
                nn.Dropout(p=dropout_1), nn.Linear(200, 100), nn.Sigmoid(),
                nn.Dropout(p=dropout_2), nn.Linear(100, 1))

    def forward(self, x):
        if self.linear:
            embedded = self.feature_embedding(x).squeeze()
        else:
            embedded = self.feature_embedding(x.transpose(-1, -2))
        min_feature, _ = torch.topk(embedded, self.retain, largest=False)
        max_feature, _ = torch.topk(embedded, self.retain)
        max_min = torch.cat([min_feature, max_feature], dim=-1)
        return self.fully_connected(max_min).flatten()

    def predict_proba(self, X):
        return self.forward(X).detach().cpu().numpy()


class Dataset(torch.utils.data.Dataset):
    """Use all data set in a tensor as it fits on RAM."""
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def fit_model(
        generator, device, validation=None, retain=2,  dropout_0=.5, dropout_1=.5, dropout_2=.275,
        learning_rate=1e-3, n_epochs=30, l2_regularization=.5, linear=True, amsgrad=False, warm_start=None):
    """Run gradient descent on data given in the generator."""
    if warm_start is None:
        model = CHOWDER(
            retain=retain,
            dropout_0=dropout_0,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
            linear=linear
        ).to(device)
        loss_function = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.FloatTensor([3.]).to(device))
    else:
        model = warm_start.to(device)
        loss_function = nn.BCEWithLogitsLoss(reduction='sum')
        learning_rate /= 5
        n_epochs //= 2

    validate = False
    if validation is not None:
        validate = True
        x_test, y_test = validation

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad)

    log_wall_time = time.time()
    tb = SummaryWriter(f'./runs/{log_wall_time}')

    iteration = 0
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_batch, y_batch in generator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device, dtype=torch.float)

            y_pred = model(x_batch)

            conv_params = torch.cat([x.view(-1) for x in model.feature_embedding.parameters()])
            l2_conv = torch.sum(conv_params ** 2)
            loss = loss_function(y_pred, y_batch) + l2_regularization * l2_conv
            epoch_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            grad = torch.cat([x.view(-1) for x in model.parameters()])
            grad_norm = torch.norm(grad)
            tb.add_scalar("Grad Norm", grad_norm, iteration)

        tb.add_scalar("Loss", epoch_loss, epoch)

        if validate:
            y_validation = model(x_test.to(device))
            score = roc_auc_score(y_test, y_validation.detach().cpu().numpy())
            tb.add_scalar("Validation AUC", score, epoch)

    tb.close()
    return model, log_wall_time
