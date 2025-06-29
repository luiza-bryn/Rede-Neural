import numpy as np
from .activation_function import activation_functions
from .loss import loss_functions
from .backpropagation import forward_phase, backward_phase
from typing import Tuple, Literal

class NeuralNetwork:
    def __init__(self, 
                 hidden_layers: Tuple[int, ...],
                 hidden_activation: list[str],
                 loss: Literal['mse', 'bce', 'cce'] = 'bce',
                 model_type: Literal['binary', 'regression', 'multiclass'] = 'binary',
                 learning_rate: float = 0.01,
                 max_iter: int = 100000,
                 random_state: int = 42) -> None:
        """
        Construtor da classe NeuralNetwork.
        """
        if len(hidden_activation) != len(hidden_layers) :
            raise ValueError(f"Número de funções de ativação ({len(hidden_activation)}) deve ser"
                              f" igual ao número de camadas ocultas ({len(hidden_layers)})")

        np.random.seed(random_state)
        self.hidden_layers = hidden_layers
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = 1e-4

        print(learning_rate)

        # Armazena a lista de nomes
        self.activation_names = hidden_activation

        # Mapeia os nomes para funções e derivadas
        self.activation_hidden = []
        self.activation_hidden_deriv = []
        for name in hidden_activation:
            func, deriv = activation_functions[name]
            self.activation_hidden.append(func)
            self.activation_hidden_deriv.append(deriv)

        if model_type == 'binary':
            self.activation_output, self.activation_output_deriv = activation_functions['sigmoid']
        elif model_type == 'regression':
            self.activation_output, self.activation_output_deriv = activation_functions['identity']
        elif model_type == 'multiclass':
            self.activation_output, self.activation_output_deriv = activation_functions['softmax']

        # Funcao de perda
        self.loss_function, self.loss_derivative = loss_functions[loss]


        # Inicializacao dos pesos e biases
        self.weights = []
        self.biases = []
      

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        _, n_features = X.shape

        # Ajuste para multiclasse
        if self.model_type == 'multiclass' and y.ndim == 1:
            num_classes = len(np.unique(y))
            y = np.eye(num_classes)[y]
            self.num_classes = num_classes

        if self.model_type == 'binary':
            output_activation_name = 'sigmoid'
        elif self.model_type == 'regression':
            output_activation_name = 'identity'
        elif self.model_type == 'multiclass':
            output_activation_name = 'softmax'

        # Definicao da estrutura de rede - camadas ocultas e camada de saida
        layer_sizes = [n_features] + list(self.hidden_layers) + [y.shape[1]]

        all_activation_names = self.activation_names + [output_activation_name]

        print(all_activation_names)

        for i in range(len(layer_sizes) - 1):
            func_name = all_activation_names[i]
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
        self.activation_funcs = self.activation_hidden + [self.activation_output]
        activation_derivatives = self.activation_hidden_deriv + [self.activation_output_deriv]

        for epoch in range(self.max_iter):
            # Forward pass
            linear_combinations, activations = forward_phase(X, self.weights, self.biases, self.activation_funcs)

            # Calculo da perda - loss
            loss_value = self.loss_function(y, activations[-1])

            # Condicao de parada
            if self.tolerance and epoch > 1 and loss_value < self.tolerance:
                break
               
            # Backward pass
            self.weights, self.biases = backward_phase(
                y, self.weights, self.biases, 
                linear_combinations,
                activations,
                activation_derivatives, 
                self.loss_derivative,
                num_layers=len(self.weights),
                learning_rate=self.learning_rate
            )

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        _, activations = forward_phase(X, self.weights, self.biases, self.activation_funcs)
        output = activations[-1]

        if self.model_type == 'binary':
            output = (output > 0.5).astype(int)
        elif self.model_type == 'multiclass':
            output = np.argmax(output, axis=1)
        elif self.model_type == 'regression':
            output = output.flatten()
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {self.model_type}")   
        
        return output