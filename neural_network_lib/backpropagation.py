import numpy as np
from typing import Callable, List, Tuple

def forward_phase(inputs: np.ndarray, 
                        weights: List[np.ndarray], 
                        biases: List[np.ndarray], 
                        activation_functions: List[Callable]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Executa a fase foward retornando as combinações lineares e ativações de todas as camadas.
    """
    activations = [inputs]
    linear_combinations = []

    current_input = inputs

    # fase de propagacao para frente
    for idx, (weight_matrix, bias_vector) in enumerate(zip(weights, biases)):
        linear_combination = np.dot(current_input, weight_matrix) + bias_vector
        linear_combinations.append(linear_combination)

        current_input = activation_functions[idx](linear_combination)
        activations.append(current_input)

    return linear_combinations, activations

def backward_phase(y: np.ndarray,
                weights: List[np.ndarray],
                biases: List[np.ndarray],
                linear_combinations: List[np.ndarray],
                activations: List[np.ndarray],
                activation_functions_derivatives: List[Callable],
                loss_derivative: Callable,
                num_layers: int,
                learning_rate: float = 0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Executa a fase backward retornando os pesos e biases atualizados.
    """
    num_samples = activations[0].shape[0]
    
    # inicializa os gradientes dos pesos e biases
    weight_gradients = [np.zeros_like(w) for w in weights]
    bias_gradients = [np.zeros_like(b) for b in biases]

    if activation_functions_derivatives[-1].__name__ == 'softmax_derivative' and loss_derivative.__name__ == 'categorical_cross_entropy_derivative':
        error_signal = activations[-1] - y
    else:
        error_signal = loss_derivative(y, activations[-1]) * activation_functions_derivatives[-1](linear_combinations[-1])

    # fase de retropropagacao
    for layer_idx in reversed(range(num_layers)):
        # calcula os gradientes dos pesos e biases
        weight_gradients[layer_idx] = np.dot(activations[layer_idx].T, error_signal ) / num_samples
        bias_gradients[layer_idx] = np.sum(error_signal , axis=0) / num_samples
        
        if layer_idx > 0:
            error_signal  = np.dot(error_signal , weights[layer_idx].T) * activation_functions_derivatives[layer_idx - 1](linear_combinations[layer_idx - 1])

    # atualiza os pesos e vieses
    for i in range(len(weights)):
        weights[i] -= learning_rate * weight_gradients[i]
        biases[i] -= learning_rate * bias_gradients[i]

    return weights, biases




