import numpy as np


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size ** 2, pattern_size ** 2))

    def train(self, patterns):
        for pattern in patterns:
            flattened_pattern = pattern.flatten()
            self.weights += np.outer(flattened_pattern, flattened_pattern)
            np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iter=100):
        flattened_pattern = pattern.flatten()
        for _ in range(max_iter):
            new_pattern = np.sign(np.dot(self.weights, flattened_pattern))
            if np.array_equal(new_pattern, flattened_pattern):
                break
            flattened_pattern = new_pattern
        return new_pattern.reshape(pattern.shape)


# Função para calcular a porcentagem de acerto entre o padrão original e o padrão recuperado
def calculate_accuracy(original_pattern, retrieved_pattern):
    total_elements = original_pattern.size
    correct_elements = np.sum(original_pattern == retrieved_pattern)
    accuracy_percentage = (correct_elements / total_elements) * 100
    return accuracy_percentage


# Padrões de treinamento
training_patterns = [
    np.array([[1, 1, 1, 1, 1],
              [1, -1, -1, -1, 1],
              [1, 1, 1, 1, 1],
              [1, -1, -1, -1, 1],
              [1, 1, 1, 1, 1]]),
    # Adicione mais padrões de treinamento conforme necessário
]

# Padrões de teste
test_patterns = [
    np.array([[1, 1, 1, 1, 1],
              [1, -1, -1, -1, 1],
              [1, 1, 1, 1, 1],
              [1, -1, -1, -1, 1],
              [1, 1, 1, 1, 1]]),
    # Adicione mais padrões de teste conforme necessário
]

# Crie e treine a rede
network = HopfieldNetwork(pattern_size=5)
network.train(training_patterns)

# Teste a recuperação dos padrões de teste
for pattern in test_patterns:
    retrieved_pattern = network.recall(pattern)
    print("Padrão Original:")
    print(pattern)
    print("\nPadrão Recuperado:")
    print(retrieved_pattern)
    accuracy = calculate_accuracy(pattern, retrieved_pattern)
    print("Porcentagem de Acerto: {:.2f}%\n".format(accuracy))
