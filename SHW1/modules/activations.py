import numpy as np
from scipy.special import erf
from scipy.special import logsumexp
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        y = np.maximum(0, input)
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        dx = (input > 0).astype(int)
        grad_input = grad_output * dx
        return grad_input


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        y = 1 / (1 + np.exp(- input))
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        sigmd = 1 / (1 + np.exp(- input))
        dx = sigmd * (1 - sigmd)
        grad_input = grad_output * dx
        return grad_input


class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        y = input * (1 / 2) * (1 + erf(input / np.sqrt(2)))
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        dx = (1 / 2) * (1 + erf(input / np.sqrt(2))) + input * (1 / np.sqrt(2 * np.pi)) * np.exp(- (input ** 2) / 2)
        grad_input = grad_output * dx
        return grad_input


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        #в задании не прописано, но на лекции было вычитание max в Softmax, поэтому сделаю его, чтобы функция была вычислительно стабильной
        exp_inp = np.exp(input - np.max(input, axis=1, keepdims=True))
        denom = np.sum(exp_inp, axis=1, keepdims=True)
        y = exp_inp / denom
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        exp_inp = np.exp(input - np.max(input, axis=1, keepdims=True))
        denom = np.sum(exp_inp, axis=1, keepdims=True)
        y = exp_inp / denom
        grad_input = y * (grad_output - np.sum(grad_output * y, axis=1, keepdims=True))
        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        #здесь в логсамэксп уже max + log(sum(exp(x_j - max)))
        y = input - logsumexp(input, axis=1, keepdims=True)
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        y = input - logsumexp(input, axis=1, keepdims=True)
        grad_input = grad_output - np.exp(y) * np.sum(grad_output, axis=1, keepdims=True)
        return grad_input
