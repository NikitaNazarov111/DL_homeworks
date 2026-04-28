import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        mse_matrix = (input - target) ** 2
        mse = np.mean(mse_matrix)
        return mse

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        B, N = input.shape
        grad_input = 2 * (input - target) / (B * N)
        return grad_input


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        log_p = self.log_softmax(input)
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        if self.label_smoothing == 0:
            indicator_matrix = np.zeros(input.shape)
            indicator_matrix[np.arange(batch_size), target] = 1
        else:
            indicator_matrix = np.full(input.shape, self.label_smoothing / num_classes)
            indicator_matrix[np.arange(batch_size), target] += (1 - self.label_smoothing)
        cel = - np.sum(log_p * indicator_matrix) / batch_size
        return cel

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        if self.label_smoothing == 0:
            indicator_matrix = np.zeros(input.shape)
            indicator_matrix[np.arange(batch_size), target] = 1
        else:
            indicator_matrix = np.full(input.shape, self.label_smoothing / num_classes)
            indicator_matrix[np.arange(batch_size), target] += (1 - self.label_smoothing)
        exp_matr = np.exp(input - np.max(input, axis=1, keepdims=True))
        sum_rows = np.sum(exp_matr, axis=1, keepdims=True)
        new_exp_matr = exp_matr / sum_rows
        grad_input = (new_exp_matr - indicator_matrix) / batch_size
        return grad_input
