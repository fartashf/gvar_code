import cPickle
import numpy as np
import os

CIFAR_DATA_DIR = '%s/datasets/cifar-10-batches-py'%os.getenv("HOME")

class Data(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def dimension(self):
        return np.prod(self.X_train.shape[1:])

    def shuffle(self):
        #train_perm = np.random.permutation(self.X_train.shape[0])
        #self.X_train, self.y_train = self.X_train[train_perm, ...], self.y_train[train_perm, ...]

        # from http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        # shuffle both arrays in-place
        rng_state = np.random.get_state()
        np.random.shuffle(self.X_train)
        np.random.set_state(rng_state)
        np.random.shuffle(self.y_train)

    @staticmethod
    def batches(X, y, mbsize):
        num_batches = X.shape[0] // mbsize
        for b in range(num_batches):
            start = b * mbsize
            end = start + mbsize

            curr_X = X[start:end, ...]
            curr_y = y[start:end, ...]
            if curr_X.ndim > 2:
                curr_X = curr_X.reshape((curr_X.shape[0], -1), order='F')
            
            if y is not None:
                yield curr_X, curr_y
            else:
                yield curr_X

    def train_batches_loop(self, X, y, mbsize, reshuffle=False):
        while True:
            for result in self.batches(self.X_train, self.y_train, mbsize):
                yield result
            if reshuffle:
                self.shuffle()
            

    def num_train_batches(self, mbsize):
        return self.X_train.shape[0] // mbsize

    def train_batches(self, mbsize):
        return self.batches(self.X_train, self.y_train, mbsize)

    def test_batches(self, mbsize):
        return self.batches(self.X_test, self.y_test, mbsize)

    def random_train_batch(self, mbsize):
        batch_id = np.random.randint(0, self.num_train_batches(mbsize))
        start = batch_id * mbsize
        end = start + mbsize
        return self.X_train[start:end, ...], self.y_train[start:end, ...]



class DataParams(object):
    def __init__(self, crop=False, normalize=True, transform=False, unit_variance=False):
        self.crop = crop
        self.normalize = normalize
        self.transform = transform
        self.unit_variance = unit_variance
        
    @classmethod
    def default(cls):
        return cls()


class DataWithTransformations(Data):
    def __init__(self, X_train_full, y_train, X_test_full, y_test):
        self.X_train_full = X_train_full
        self.y_train = y_train
        self.y_train_orig = self.y_train.copy()
        self.X_test_full = X_test_full
        self.y_test = y_test

        Xtf = X_test_full.reshape((-1, 32, 32, 3), order='F')
        Xtf = Xtf[:, 2:-2, 2:-2, :]
        self.X_test = Xtf.reshape((-1, 28*28*3), order='F')

        self.shuffle()

    def shuffle(self):
        """In addition to shuffling, also re-sample the transformations."""
        ntrain = self.X_train_full.shape[0]
        self.X_train = np.zeros((ntrain, 28*28*3))
        self.y_train = self.y_train_orig.copy()
        Xtf = self.X_train_full.reshape((ntrain, 32, 32, 3), order='F')

        for k in range(ntrain):
            istart = np.random.randint(0, 5)
            jstart = np.random.randint(0, 5)
            patch = Xtf[k, istart:istart+28, jstart:jstart+28, :]
            if np.random.binomial(1, 0.5):
                patch = patch[::-1, :, :]
            self.X_train[k, :] = patch.reshape((28 * 28 * 3), order='F')

        super(DataWithTransformations, self).shuffle()



def cifar_train_batch_path(batch_id):
    return os.path.join(CIFAR_DATA_DIR, 'data_batch_{}'.format(batch_id))
def cifar_test_batch_path():
    return os.path.join(CIFAR_DATA_DIR, 'test_batch')


def load_data(params):
    X_train = np.zeros((0, 32*32*3), dtype=float)
    y_train = np.zeros((0,), dtype=int)
    for batch_id in range(1, 6):
        train_obj = cPickle.load(open(cifar_train_batch_path(batch_id)))
        X_train = np.vstack([X_train, train_obj['data'].astype(float)])
        y_train = np.concatenate([y_train, np.array(train_obj['labels'])])

    ## X_train_ = X_train.reshape((-1, 32, 32, 3), order='F')
    ## X_train_ = (X_train_ - X_train_.min()) / (X_train_.max() - X_train_.min())
    ## from matplotlib import pyplot
    ## fig, axes = pyplot.subplots(10, 10)
    ## for i, ax in enumerate(axes.ravel()):
    ##     ax.imshow(X_train_[i, :, :, :])
    ## fig.canvas.draw()

    
    test_obj = cPickle.load(open(cifar_test_batch_path()))
    X_test = test_obj['data'].astype(float)
    y_test = np.array(test_obj['labels'])

    if params.crop:
        assert not params.transform
        X_train = X_train.reshape((-1, 32, 32, 3), order='F')
        X_train = X_train[:, 2:-2, 2:-2, :]
        X_train = X_train.reshape((-1, 28*28*3), order='F')

        X_test = X_test.reshape((-1, 32, 32, 3), order='F')
        X_test = X_test[:, 2:-2, 2:-2, :]
        X_test = X_test.reshape((-1, 28*28*3), order='F')



    # normalize to zero mean, unit variance
    if params.normalize:
        mean = X_train.mean(0)
        X_train -= mean
        X_test -= mean

        if params.unit_variance:
            std = X_train.std()
            X_train /= std
            X_test /= std

    if params.transform:
        return DataWithTransformations(X_train, y_train, X_test, y_test)
    else:
        return Data(X_train, y_train, X_test, y_test)
