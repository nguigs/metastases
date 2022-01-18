import numpy as np
import pathlib
import torch
from chowder import CHOWDER
from submission_tools import get_train_set, get_test_set, save_predictions

data_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/data')
pred_dir = pathlib.Path('/user/nguigui/home/PycharmProjects/metastases/predictions')
model = CHOWDER()

X_train, y_train = get_train_set(data_dir)
X_test, ids_test = get_test_set(data_dir)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
save_predictions(data_dir, 'dummy', ids_test, model, X_test)
