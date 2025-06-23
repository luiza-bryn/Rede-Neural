import numpy as np
from activation_function import activation_functions
from loss import loss_functions
from backpropagation import forward_phase, backward_phase
from typing import Tuple

class NeuralNetwork:
    def __init__(self, 
                 hidden_layers: Tuple[int, ...],
                 activation: str,
                 loss: str,
                 model_type: str,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 random_state: int = 42) -> None:
        """
        Construtor da classe NeuralNetwork.
        """
        np.random.seed(random_state)
        self.hidden_layers = hidden_layers
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.activation_function, self.activation_function_derivative = activation_functions[activation]
        self.loss_function, self.loss_derivative = loss_functions[loss]

        self.weights = []
        self.biases = []


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        pass