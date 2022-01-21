import numpy as np
import pathlib
import torch
from sklearn.metrics import roc_auc_score
from chowder import fit_model, Dataset
from submission_tools import get_train_set, get_test_set, save_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')
pred_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/predictions')

X_train, y_train = get_train_set(data_dir)
X_train = torch.from_numpy(
    X_train.astype(np.float32)#.reshape((-1, 2048, 1000))
)

X_test, ids_test = get_test_set(data_dir)
X_test = torch.from_numpy(
    X_test.astype(np.float32)#.reshape((-1, 2048, 1000))
)

n_runs = 10
hyperparams = {
    'retain': 5,
    'dropout_0': .1,
    'dropout_1': .5,
    'dropout_2': .3,
    'learning_rate': 1e-3,
    'n_epochs': 30,
    'l2_regularization': .5,
    'linear': True
}

training_set = Dataset(X_train, y_train)
generator = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

train_preds = []
test_preds = []
for _ in range(n_runs):
    model = fit_model(generator, device, **hyperparams).cpu()

    y_pred_train = model.predict_proba(X_train)
    train_preds.append(y_pred_train)

    y_pred_test = model.predict_proba(X_test)
    test_preds.append(y_pred_test)

mean_train_pred = np.stack(train_preds).mean(axis=0)
print(roc_auc_score(y_train, mean_train_pred))

mean_test_pred = np.stack(test_preds).mean(axis=0)
save_predictions(pred_dir, 'paper', ids_test, mean_test_pred)
