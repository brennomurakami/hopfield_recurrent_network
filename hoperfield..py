import numpy as np
import time
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        # Inicialização da matriz de pesos com zeros
        self.weights = np.zeros((pattern_size ** 2, pattern_size ** 2))

    def train(self, patterns):
        start_time = time.time()
        for pattern in patterns:
            flattened_pattern = pattern.flatten()
            # Cálculo da matriz de pesos: soma externa dos padrões
            self.weights += np.outer(flattened_pattern, flattened_pattern)
            # Atribuição de zeros na diagonal da matriz de pesos
            np.fill_diagonal(self.weights, 0)
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def recall(self, pattern, max_iter=100):
        flattened_pattern = pattern.flatten()
        num_iterations = 0
        for _ in range(max_iter):
            num_iterations += 1
            # Atualização do padrão de forma iterativa
            new_pattern = np.sign(np.dot(self.weights, flattened_pattern))
            # Verificação se o padrão convergiu
            if np.array_equal(new_pattern, flattened_pattern):
                break
            flattened_pattern = new_pattern
        return new_pattern.reshape(pattern.shape), num_iterations


# Função para calcular a porcentagem de acerto entre o padrão original e o padrão recuperado
def calculate_accuracy(original_pattern, retrieved_pattern):
    original_pattern_int = original_pattern.astype(int)
    retrieved_pattern_int = retrieved_pattern.astype(int)

    total_elements = original_pattern_int.size
    correct_elements = np.sum(original_pattern_int == retrieved_pattern_int)
    accuracy_percentage = (correct_elements / total_elements) * 100
    return accuracy_percentage


def visualize_matrices(original_matrix, recovered_matrix):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original_matrix, cmap='Greens', interpolation='nearest')
    axs[0].set_title("Padrão de Teste")
    axs[0].axis('off')
    axs[1].imshow(recovered_matrix, cmap='Greens', interpolation='nearest')
    axs[1].set_title("Padrão Recuperado")
    axs[1].axis('off')
    plt.show()


# Padrões de treinamento
training_patterns = [
    # E
    np.array([[1,  1,  1,  1,  1],
              [1, -1, -1, -1, -1],
              [1,  1,  1,  1,  1],
              [1, -1, -1, -1, -1],
              [1,  1,  1,  1,  1]]),
    # Y
    np.array([[1, -1, -1, -1,  1],
              [-1, 1, -1,  1, -1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1],
              [-1, -1, 1, -1, -1]]),
    # N
    np.array([[1, -1, -1, -1, 1],
              [1,  1, -1, -1, 1],
              [1, -1,  1, -1, 1],
              [1, -1, -1,  1, 1],
              [1, -1, -1, -1, 1]]),
    # IF
    np.array([[1, -1, 1,  1,  1],
             [-1, -1, 1, -1, -1],
              [1, -1, 1,  1,  1],
              [1, -1, 1, -1, -1],
              [1, -1, 1, -1, -1]]),
]

# Padrões de teste
test_patterns = [
    # E falhado
    np.array([[1,  1, -1,  1,  1],
             [-1, -1, -1, -1, -1],
              [1,  1, -1, -1, -1],
              [1, -1, -1, -1, -1],
              [1,  1,  1,  1, -1]]),
    # Y falhado
    np.array([[1,  1, -1, -1,  1],
             [-1,  1, -1,  1, -1],
             [-1, -1,  1, -1, -1],
             [-1, -1,  1, -1, -1],
             [-1,  1, -1, -1, -1]]),
    # M
    np.array([[1, -1, -1, -1, 1],
              [1,  1, -1,  1, 1],
              [1, -1,  1, -1, 1],
              [1, -1, -1, -1, 1],
              [1, -1, -1, -1, 1]]),
    # IF falhado
    np.array([[-1, -1, 1, -1,  1],
              [1,  -1, 1, -1, -1],
             [-1,  -1, -1,  1,  1],
              [1,  -1, 1, -1, -1],
              [1,  -1, 1, -1, -1]]),
]

# Crie e treine a rede
network = HopfieldNetwork(pattern_size=5)
training_time = network.train(training_patterns)
num_iteracoes_totais = 0

# Teste a recuperação dos padrões de teste
for pattern in test_patterns:
    retrieved_pattern, num_iterations = network.recall(pattern)
    num_iteracoes_totais+= num_iterations
    print("Padrão Teste:")
    print(pattern)
    print("\nPadrão Recuperado:")
    print(retrieved_pattern)
    accuracy = calculate_accuracy(pattern, retrieved_pattern)
    print("Porcentagem de Semelhança do Padrão de Teste com o padrão recuperado: {:.2f}%\n".format(accuracy))
    print("Número de iterações:", num_iterations)
    visualize_matrices(pattern, retrieved_pattern)

media_iteracoes_por_teste = num_iteracoes_totais / len(test_patterns)
media_tempo_recuperacao = training_time / len(test_patterns)

print("Estatísticas gerais")
print("Tempo de Treinamento: {:.6f} segundos".format(training_time))
print("Número de iterações totais:", num_iteracoes_totais)
print("Média de iterações por teste:", media_iteracoes_por_teste)
print("Tempo médio de recuperação por teste: {:.6f} segundos".format(media_tempo_recuperacao))
