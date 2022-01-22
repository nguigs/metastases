import json
import logging

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

log_key = round(time())
logging.basicConfig(filename=pred_dir / f'log_{log_key}.log', level=logging.INFO)

n_runs = 40
hyperparams = {
    'retain': 5,
    'dropout_0': .2,
    'dropout_1': .5,
    'dropout_2': .3,
    'learning_rate': 1e-3,
    'n_epochs': 60,
    'l2_regularization': .5,
    'linear': True,
    'amsgrad': True
}
logging.info(json.dumps(hyperparams))

training_set = Dataset(X_train, y_train)
generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

train_preds = []
test_preds = []
model = None
for i in range(n_runs):
    warm_start = model if i % 2 == 1 else None
    model, log_time = fit_model(generator, device, warm_start=warm_start, **hyperparams)
    model = model.cpu()

    log_key = round(log_time)

    y_pred_train = model.predict_proba(X_train)
    train_preds.append(y_pred_train)
    training_score = roc_auc_score(y_train, y_pred_train)
    logging.info(f'run {i} with log {log_time}, train_auc: {training_score}')

    if training_score < .7:  # happens
        logging.info(f'run {i} with log {log_time} thrown away')
        continue

    y_pred_test = model.predict_proba(X_test)
    test_preds.append(y_pred_test)

mean_train_pred = np.stack(train_preds).mean(axis=0)
logging.info(f'Score of bagged predictions : {roc_auc_score(y_train, mean_train_pred)}')

mean_test_pred = np.stack(test_preds).mean(axis=0)
save_predictions(pred_dir, f'paper_{log_key}', ids_test, mean_test_pred)
