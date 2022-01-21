import json

import numpy as np
import pathlib
import torch
from time import time
from sklearn.metrics import roc_auc_score
from chowder import fit_model, Dataset
from submission_tools import get_train_set, get_test_set, save_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')
pred_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/predictions')

X_train, y_train, n_tiles = get_train_set(data_dir)
X_train = torch.from_numpy(
    X_train.astype(np.float32)#.reshape((-1, 2048, 1000))
)

X_test, ids_test, n_tiles_test = get_test_set(data_dir)
X_test = torch.from_numpy(
    X_test.astype(np.float32)#.reshape((-1, 2048, 1000))
)

n_runs = 10
hyperparams = {
    'retain': 10,
    'dropout_0': .5,
    'dropout_1': .5,
    'dropout_2': .5,
    'learning_rate': 5e-4,
    'n_epochs': 80,
    'l2_regularization': 3.,
    'linear': True,
    'amsgrad': True
}

training_set = Dataset(X_train, y_train)
generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

train_preds = []
test_preds = []
for i in range(n_runs):
    model, _ = fit_model(generator, device, **hyperparams)
    model = model.cpu()

    y_pred_train = model.predict_proba(X_train)
    train_preds.append(y_pred_train)

    if roc_auc_score(y_train, y_pred_train) < .7:  # this happens
        print(f'run {i} thrown away')
        continue

    y_pred_test = model.predict_proba(X_test)
    test_preds.append(y_pred_test)

mean_train_pred = np.stack(train_preds).mean(axis=0)
print(roc_auc_score(y_train, mean_train_pred))

mean_test_pred = np.stack(test_preds).mean(axis=0)
log_key = round(time())
save_predictions(pred_dir, f'paper_{log_key}', ids_test, mean_test_pred)
with open(pred_dir / f'log_{log_key}', 'a') as file:
    file.write(json.dumps(hyperparams))
