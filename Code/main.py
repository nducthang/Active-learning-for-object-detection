
from sklearn.datasets import fetch_mldata
from sklearn.utils import check_random_state
import numpy as np

trainset_size = 60000  # ie., testset_size = 10000
max_queried = 500


# ==============================================================================
def download():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target
    print('MNIST:', X.shape, y.shape)
    return (X, y)


(X, y) = download()


def split(train_size):
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_full, y_train_full, X_test, y_test)


(X_train_full, y_train_full, X_test, y_test) = split(trainset_size)

print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)


def get_k_random_samples(initial_labeled_samples, X_train_full,
                         y_train_full):
    random_state = check_random_state(0)
    permutation = np.random.choice(trainset_size,
                                   initial_labeled_samples,
                                   replace=False)
    print()
    print('initial random chosen samples', permutation.shape)
    X_train = X_train_full[permutation]
    y_train = y_train_full[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    print(
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
    )
    return (permutation, X_train, y_train)
