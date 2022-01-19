import numpy as np
import pathlib
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from chowder import Dataset, fit_model
from submission_tools import get_train_set

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')

X_train, y_train = get_train_set(data_dir)
X_train = torch.from_numpy(X_train.astype(np.float32).reshape((-1, 2048, 1000)))

n_folds = 3
n_runs = 1
hyperparams = {
    'retain': 5,
    'dropout_0': 0,
    'dropout_1': .5,
    'dropout_2': .3,
    'learning_rate': 1e-3,
    'n_epochs': 30,
    'l2_regularization': .2
}

cv = StratifiedKFold(n_folds, shuffle=True)

train_preds = []
test_preds = []
for train_idx, test_idx in cv.split(X_train, y_train):
    for _ in range(n_runs):
        training_set = Dataset(X_train[train_idx], y_train[train_idx])
        generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)
        model = fit_model(generator, device, **hyperparams).cpu()

        y_pred_train = model.predict_proba(X_train[train_idx])
        train_preds.append(y_pred_train)

        y_pred_test = model.predict_proba(X_train[test_idx])
        test_preds.append(y_pred_test)

    y_pred_mean_train = np.mean(train_preds, axis=0)
    y_pred_mean_test = np.mean(test_preds, axis=0)

    print('train:', roc_auc_score(y_train[train_idx], y_pred_mean_train))
    print('test:', roc_auc_score(y_train[test_idx], y_pred_mean_test))
