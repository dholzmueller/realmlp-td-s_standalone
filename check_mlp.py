import numpy as np
import torch
import pandas as pd
import time

from sklearn.model_selection import train_test_split

from mlp import Standalone_RealMLP_TD_S_Classifier, Standalone_RealMLP_TD_S_Regressor
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_S_Classifier, RealMLP_TD_S_Regressor


def get_dataset(n_samples: int, classification: bool):
    np.random.seed(0)
    x = np.random.randn(n_samples, 3) ** 3
    x_ds = pd.DataFrame(x, columns=['a', 'b', 'c'])
    x_ds['cat_a'] = np.random.randint(3, size=(n_samples,))
    x_ds['cat_b'] = np.random.randint(4, size=(n_samples,))
    x_ds.loc[x_ds['cat_a'] == 0, 'cat_a'] = np.NaN
    x_ds.loc[x_ds['cat_b'] == 0, 'cat_b'] = np.NaN
    x_ds['cat_a'] = x_ds['cat_a'].astype('category')
    x_ds['cat_b'] = x_ds['cat_b'].astype('category')

    print(x_ds.head())

    if classification:
        y = np.array(np.fmod(x[:, 0], 1) > 0.5, dtype=np.int32)
    else:
        y = x[:, 0]
    return x_ds, y


def check_equality(classification: bool):
    np.random.seed(1)
    torch.manual_seed(1)

    n_train = 512
    n_valid = 20
    n_test = 300
    n_samples = n_train + n_valid + n_test

    x, y = get_dataset(n_samples, classification=classification)

    x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=n_test, random_state=0)

    trainval_idxs = np.arange(n_train + n_valid)
    x_train, x_val, y_train, y_val, train_idxs, val_idxs = train_test_split(x_trainval, y_trainval,
                                                                            trainval_idxs,
                                                                            test_size=n_valid, random_state=0)

    # print(f'{x_train=}')
    # print(f'{y_train=}')
    # print(f'{train_idxs=}')
    # print(f'{x_val=}')
    # print(f'{y_val=}')
    # print(f'{val_idxs=}')

    n_repeats = 10

    preds_1 = []
    preds_2 = []
    preds_3 = []

    seed_offset=1000

    start_time = time.time()

    for i in range(n_repeats):
        print(f'Round {i}')
        seed = i + seed_offset
        np.random.seed(seed)
        torch.manual_seed(seed)
        if classification:
            mlp1 = Standalone_RealMLP_TD_S_Classifier()
            mlp2 = RealMLP_TD_S_Classifier(device='cpu', random_state=seed)
            mlp3 = RealMLP_TD_S_Classifier(device='cpu', random_state=seed, lr=5e-2)
        else:
            mlp1 = Standalone_RealMLP_TD_S_Regressor()
            mlp2 = RealMLP_TD_S_Regressor(device='cpu', random_state=seed)
            mlp3 = RealMLP_TD_S_Regressor(device='cpu', random_state=seed, lr=5e-2)
        mlp1.fit(x_train, y_train, x_val, y_val)
        mlp2.fit(x_trainval, y_trainval, val_idxs=val_idxs)
        mlp3.fit(x_trainval, y_trainval, val_idxs=val_idxs)
        if classification:
            preds_1.append(mlp1.predict_proba(x_test)[:, 1])
            preds_2.append(mlp2.predict_proba(x_test)[:, 1])
            preds_3.append(mlp3.predict_proba(x_test)[:, 1])
        else:
            preds_1.append(mlp1.predict(x_test))
            preds_2.append(mlp2.predict(x_test))
            preds_3.append(mlp3.predict(x_test))
        print(f'{preds_1[-1].shape=}')
        # preds_1.append(mlp1.predict_proba(x_train))
        # preds_2.append(mlp2.predict_proba(x_train))
        # print(f'{mlp1.classes_=}, {mlp2.classes_=}')

    preds_1 = np.asarray(preds_1)
    preds_2 = np.asarray(preds_2)
    preds_3 = np.asarray(preds_3)

    # for i, arr in enumerate([preds_1, preds_2, preds_3]):
    #     # arr = arr.reshape(arr.shape[0], -1)
    #     print(
    #         f'Predictions for model {i + 1}: {np.mean(arr, axis=0)} +- {(2 / np.sqrt(n_repeats - 1)) * np.std(arr, axis=0)}')

    print(f'Comparing models 1, 2 (which should be equal), and 3 (which is different)')
    print(f'1-2 should get around 95% within the interval, 1-3 and 2-3 should get less')

    for label, arr1, arr2 in [('1-2', preds_1, preds_2), ('1-3', preds_1, preds_3), ('2-3', preds_2, preds_3)]:
        positives = np.sum(np.abs(np.mean(arr1, axis=0)-np.mean(arr2, axis=0)) <=
                        (1/np.sqrt(n_repeats-1)) * (np.std(arr1, axis=0) + np.std(arr2, axis=0)))
        print(f'Number of {label} test predictions within apx 95% intervals: {positives}/{n_test} = {positives/n_test*100:g}%')

    print(f'Total time: {time.time() - start_time:g} s')


if __name__ == '__main__':
    check_equality(classification=True)
    # check_equality(classification=False)
    pass

