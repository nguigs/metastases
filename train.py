import numpy as np
import pathlib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from chowder import Dataset, fit_model
from submission_tools import get_train_set

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')

X_train, y_train, n_tiles = get_train_set(data_dir)
X_train = torch.from_numpy(
    X_train.astype(np.float32)#.reshape((-1, 2048, 1000))
)

n_folds = 5
n_runs = 2
hyperparams = {
    'retain': 5,
    'dropout_0': .2,
    'dropout_1': .5,
    'dropout_2': .3,
    'learning_rate': 1e-3,
    'n_epochs': 30,
    'l2_regularization': .5,
    'linear': True,
    'amsgrad': True
}

cv = StratifiedKFold(n_folds, shuffle=True, random_state=1)

roc_test = []
j = 0
for train_idx, test_idx in cv.split(X_train, y_train):
    train_preds = []
    test_preds = []
    print(f'Fold {j} --------------------------')

    estimator = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    estimator.fit(X_train[train_idx].mean(1), y_train[train_idx])
    baseline_pred = estimator.predict_proba(X_train[test_idx].mean(1))[:, 1]
    validation = (X_train[test_idx], y_train[test_idx])
    model = None
    for i in range(n_runs):
        warm_start = model if i % 2 == 1 else None

        training_set = Dataset(X_train[train_idx], y_train[train_idx])
        generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

        model, log_time = fit_model(
            generator, device, validation, warm_start=warm_start, **hyperparams)

        model = model.cpu()
        y_pred_train = model.predict_proba(X_train[train_idx])
        train_preds.append(y_pred_train)

        y_pred_test = model.predict_proba(X_train[test_idx])
        test_preds.append(y_pred_test)

        print(f'run {i} -- {loss} -- {warm_start is None} -- {log_time}-----------------------------')
        print('train:', roc_auc_score(y_train[train_idx], y_pred_train))
        score = roc_auc_score(y_train[test_idx], y_pred_test)
        print('test:', score)
        print('baseline test:', roc_auc_score(y_train[test_idx], baseline_pred))

    # append baseline pred for very bad cases
    test_preds.append(baseline_pred)

    y_pred_mean_train = np.mean(train_preds, axis=0)
    y_pred_mean_test = np.mean(test_preds, axis=0)

    print('mean runs ---------------------------------')
    print('train:', roc_auc_score(y_train[train_idx], y_pred_mean_train))
    score = roc_auc_score(y_train[test_idx], y_pred_mean_test)
    roc_test.append(score)
    print('test:', score)
    j += 1

aucs = np.array(roc_test)
print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))
