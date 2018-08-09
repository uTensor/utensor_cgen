# -*- coding: utf8 -*-
from __future__ import print_function
from __future__ import absolute_import
import os
import tarfile

import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from .dataset import DataSet, dense_to_one_hot
from .cs231n.data_utils import load_CIFAR10

__all__ = ["read_data_sets", "get_class_names", "onehot_to_names"]

_SOURCE_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_LABELS_MAP = {0: 'plane', 1: 'car', 2: 'bird',
               3: 'cat', 4: 'deer', 5: 'dog',
               6: 'frog', 7: 'horse', 8: 'ship',
               9: 'truck'}


def read_data_sets(work_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=None,
                   seed=None):
    if fake_data:
        def fake():
            return DataSet([], [],
                           fake_data=True,
                           image_dims=32*32*3,
                           num_class=10,
                           one_hot=one_hot,
                           dtype=dtype,
                           seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    root_data_dir = os.path.join(work_dir, "cifar-10-batches-py")
    if not os.path.exists(root_data_dir):
        # no data directory found
        # download gz file
        print("Trying to download cifar data (if the tar.gz file is not available)")
        gz_fpath = base.maybe_download("cifar-10-python.tar.gz",
                                       work_dir,
                                       _SOURCE_URL)
        print("Extracting data in {}".format(root_data_dir))
        with tarfile.open(gz_fpath) as tar:
            tar.extractall(work_dir)
    else:
        print("cifar data directory found {}".format(root_data_dir))
    print("loading data...")
    X_train, Y_train, X_test, Y_test = load_CIFAR10(root_data_dir)
    if one_hot:
        num_class_train = len(np.unique(Y_train))
        num_class_test = len(np.unique(Y_test))
        assert num_class_test == num_class_train, \
            "number of classes mismatch: {} and {}".format(num_class_train, num_class_test)
        Y_train = dense_to_one_hot(Y_train, num_class_train)
        Y_test = dense_to_one_hot(Y_test, num_class_test)
    if validation_size is None:
        validation_size = int(X_train.shape[0]/10)
    valid_idx = np.random.choice(range(X_train.shape[0]), validation_size)
    mask = np.array([True if row_idx in valid_idx else False for row_idx in range(X_train.shape[0])])
    X_train, X_valid = X_train[~mask], X_train[mask]
    Y_train, Y_valid = Y_train[~mask], Y_train[mask]

    train_dataset = DataSet(X_train, Y_train,
                            one_hot=one_hot,
                            dtype=dtype,
                            reshape=reshape,
                            seed=seed)
    valid_dataset = DataSet(X_valid, Y_valid,
                            one_hot=one_hot,
                            dtype=dtype,
                            reshape=reshape,
                            seed=seed)
    test_dataset = DataSet(X_test, Y_test,
                           one_hot=one_hot,
                           dtype=dtype,
                           reshape=reshape,
                           seed=seed)
    return base.Datasets(train=train_dataset,
                         validation=valid_dataset,
                         test=test_dataset)


def get_class_names(labels):
    return np.vectorize(_LABELS_MAP.get)(labels)


def onehot_to_names(one_hot):
    labels = np.argmax(one_hot, axis=1)
    return get_class_names(labels)
