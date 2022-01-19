import os
import numpy as np
import pandas as pd


def get_features(filenames):
    """Load the resnet features and add zeros for missing tiles.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 1000, 2048)`
    """
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f)

        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]
        n_tiles = len(patient_features)
        if n_tiles < 1000:
            patient_features = np.concatenate([patient_features, np.zeros((1000 - n_tiles, 2048))])

        features.append(patient_features)

    features = np.stack(features, axis=0)
    return features


def get_train_set(data_dir):
    assert data_dir.is_dir()
    train_dir = data_dir / "train_input" / "resnet_features"
    train_output_filename = data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir / k for k in os.listdir(train_dir)]
    index = []
    for filename in filenames_train:
        pid = filename.stem.split('_')[1]
        index.append(int(pid))
        assert filename.is_file(), filename
    index_sorted = np.argsort(index)

    # Get the features for training
    features_train = get_features(np.stack(filenames_train)[index_sorted])

    # Get the labels
    labels_train = train_output["Target"].values
    return features_train, labels_train


def get_test_set(data_dir):
    assert data_dir.is_dir()
    test_dir = data_dir / "test_input" / "resnet_features"

    # Get the filenames for train
    filenames_test = [test_dir / k for k in os.listdir(test_dir)]
    index = []
    for filename in filenames_test:
        pid = filename.stem.split('_')[1]
        index.append(int(pid))
        assert filename.is_file(), filename
    index_sorter = np.argsort(index)

    # Get the features for training
    features_test = get_features(np.array(filenames_test)[index_sorter])
    return features_test, [f"ID_{k}" for k in np.array(index)[index_sorter]]


def save_predictions(pred_dir, name, ids_test, predictions):
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": predictions})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(pred_dir / f"preds_test_{name}.csv")
