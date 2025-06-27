import numpy as np
from activation_function import activation_functions
from loss import loss_functions
from backpropagation import forward_phase, backward_phase
from typing import Tuple

class NeuralNetwork:
    def __init__(self, 
                 hidden_layers: Tuple[int, ...],
                 activation: list[str],
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
        self.tolerance = 1e-4

        # Armazena a lsita de nomes
        self.activation_names = activation

        # Mapeia os nomea para funcoes e derivadas
        self.activation_hidden, self.activation_hidden_deriv = activation_functions[activation[0]]
        self.activation_output, self.activation_output_deriv = activation_functions[activation[1]]

        self.activation_hidden_name = activation[0]
        self.activation_output_name = activation[1]

        # Funcao de perda
        self.loss_function, self.loss_derivative = loss_functions[loss]


        # Inicializacao dos pesos e biases
        self.weights = []
        self.biases = []
      

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        _, n_features = X.shape

        # Definicao da estrutura de rede - camadas ocultas e camada de saida
        layer_sizes = [n_features] + list(self.hidden_layers) + [y.shape[1]]

        for i in range(len(layer_sizes) - 1):
            print(i)
            print(self.activation_names[0])
            func_name = self.activation_names[0]
            if func_name == 'relu':
                scale = np.sqrt(2 / layer_sizes[i])
            elif func_name in ['tanh', 'sigmoid']:
                scale = np.sqrt(1 / layer_sizes[i])
            else:
                scale = 1
    
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            bias_vector = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

        # Funcoes de ativacao por camada
        self.activation_funcs = [self.activation_hidden] * len(self.hidden_layers) + [self.activation_output]
        activation_derivatives = [self.activation_hidden_deriv] * len(self.hidden_layers) + [self.activation_output_deriv]

        for epoch in range(self.max_iter):
            # Forward pass
            _, activations = forward_phase(X, self.weights, self.biases, self.activation_funcs)

            # Calculo da perda - loss
            loss_value = self.loss_function(y, activations[-1])

            # Condicao de parada
            if self.tolerance and epoch > 1 and loss_value < self.tolerance:
                break

            # Backward pass
            self.weights, self.biases = backward_phase(
                X, y, self.weights, self.biases, 
                self.activation_funcs,
                activation_derivatives, 
                self.loss_derivative,
                num_layers=len(self.weights),
                learning_rate=self.learning_rate
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        _, activations = forward_phase(X, self.weights, self.biases, self.activation_funcs)
        output = activations[-1]

        if self.model_type == 'classification':
            if self.loss_function == 'categorical_crossentropy':
                return np.argmax(output, axis=1) # Multiclasse
            return (output > 0.5).astype(int) # Binaria       
        
        return output