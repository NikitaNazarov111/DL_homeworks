import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        if self.y.shape[0] % self.batch_size == 0:
            return self.y.shape[0] // self.batch_size
        else:
            return self.y.shape[0] // self.batch_size + 1

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return self.y.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        self.batch_id = 0
        indexes = np.arange(self.y.shape[0])
        if self.shuffle:
            np.random.shuffle(indexes)
        self.indexes = indexes
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        if self.batch_id >= self.__len__():
            raise StopIteration
        else:
            st = self.batch_size * self.batch_id
            fin = st + self.batch_size
            curr_batch_ind = self.indexes[st : fin]
            x_batch = self.X[curr_batch_ind]
            y_batch = self.y[curr_batch_ind]
            self.batch_id += 1
            return (x_batch, y_batch)
