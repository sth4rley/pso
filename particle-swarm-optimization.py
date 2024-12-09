import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import math

matplotlib.use('tkAgg')

# Classe para Partícula
class Particula:
    def __init__(self, fitness_function, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.posicao = [0.0] * dim
        self.velocidade = [0.0] * dim
        self.melhor_pos_part = [0.0] * dim

        # Inicializa a posição e velocidade da partícula aleatoriamente
        for i in range(dim):
            self.posicao[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocidade[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness_function = fitness_function
        self.fitness = self.fitness_function(self.posicao)
        self.melhor_pos_part = copy.copy(self.posicao)
        self.melhor_fitness_part = self.fitness

# PSO com animação
def pso(fitness_function, max_iter, n, dim, minx, maxx):
    # Hiperparâmetros
    inercia = 0.729
    fator_cognitivo = 1.49445
    fator_social = 1.49445

    rnd = random.Random(0)

    # Cria n partículas aleatórias
    enxame = [Particula(fitness_function, dim, minx, maxx, i) for i in range(n)]

    # Melhor solução global
    melhor_pos_enxame = [0.0 for _ in range(dim)]
    melhor_fitness_enxame = float('inf')

    # Configurações do gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(minx, maxx, 100)
    y = np.linspace(minx, maxx, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[fitness_function([xi, yi]) for xi in x] for yi in y])
    contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")

    # Inicializa os pontos das partículas e da melhor solução
    particles_scatter = ax.scatter([], [], color="blue", label="Partículas")
    best_scatter = ax.scatter([], [], color="red", label="Melhor solução")
    ax.legend()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(minx, maxx)
    ax.set_title("PSO: Otimização Animada")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(contour, label="Fitness")

    # Função para atualizar o gráfico em cada frame
    def update(frame):
        nonlocal melhor_fitness_enxame, melhor_pos_enxame

        # Atualiza cada partícula
        for i in range(n):
            # Calcula a nova velocidade da partícula
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()

                # Atualiza a velocidade
                # v(t+1) = w * v(t) + c1 * r1 * (p(t) - x(t)) + c2 * r2 * (g(t) - x(t))
                # ou seja, a nova velocidade é a soma de três componentes:
                # 1. A inércia da velocidade atual
                # 2. A atração cognitiva (p(t) - x(t)), onde p(t) é a melhor posição da partícula e x(t) é a posição atual
                # 3. A atração social (g(t) - x(t)), onde g(t) é a melhor posição do enxame
                
                enxame[i].velocidade[k] = (
                    inercia * enxame[i].velocidade[k] +
                    fator_cognitivo * r1 * (enxame[i].melhor_pos_part[k] - enxame[i].posicao[k]) +
                    fator_social * r2 * (melhor_pos_enxame[k] - enxame[i].posicao[k])
                )

                # Limita a velocidade
                enxame[i].velocidade[k] = max(minx, min(enxame[i].velocidade[k], maxx))

            # Atualiza a posição da partícula
            for k in range(dim):
                enxame[i].posicao[k] += enxame[i].velocidade[k]

            # Calcula o fitness da nova posição
            enxame[i].fitness = fitness_function(enxame[i].posicao)

            # Atualiza a melhor posição da partícula
            if enxame[i].fitness < enxame[i].melhor_fitness_part:
                enxame[i].melhor_fitness_part = enxame[i].fitness
                enxame[i].melhor_pos_part = copy.copy(enxame[i].posicao)

            # Atualiza a melhor solução global
            if enxame[i].fitness < melhor_fitness_enxame:
                melhor_fitness_enxame = enxame[i].fitness
                melhor_pos_enxame = copy.copy(enxame[i].posicao)

        # Atualiza os dados do gráfico
        x_data = [p.posicao[0] for p in enxame]
        y_data = [p.posicao[1] for p in enxame]
        particles_scatter.set_offsets(np.c_[x_data, y_data])
        best_scatter.set_offsets(np.c_[melhor_pos_enxame[0], melhor_pos_enxame[1]])
        ax.set_title(f"Iteração {frame + 1} - Melhor Fitness = {melhor_fitness_enxame:.6f}")

    # Configura a animação
    ani = FuncAnimation(fig, update, frames=max_iter, interval=75, repeat=False)

    plt.show()

    return melhor_pos_enxame

# Funções de fitness

# Função Quadrática Simples (Parabólica) 
# f(x, y) = x^2 + y^2 
def fitness_quadratica(pos):
    x, y = pos
    return x**2 + y**2

# Função Griewank
# f(x, y) = 1 + (x^2 + y^2) / 4000 - cos(x) * cos(y / sqrt(2)) 
def fitness_griewank(pos):
    x, y = pos
    return 1 + (x**2 + y**2) / 4000 - math.cos(x) * math.cos(y / math.sqrt(2))

# Função de Ackley
# f(x, y) = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2 * π * x) + cos(2 * π * y))) + e + 20
def fitness_ackley(pos):
    x, y = pos
    return -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20

# Função de Schaffer
# f(x, y) = 0.5 + (sin(x^2 - y^2)^2 - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
def fitness_schaffer(pos):
    x, y = pos
    return 0.5 + (math.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

# Função de Rastrigin 
# f(x) = 10 * 2 + sum([(x^2 - 10 * cos(2 * π * x))]) 
def fitness_rastrigin(pos):
    return 10 * len(pos) + sum([x**2 - 10 * math.cos(2 * math.pi * x) for x in pos])


# Executa o programa principal
if __name__ == "__main__":
    # Escolha da função de fitness
    melhor_solucao = pso(fitness_quadratica, max_iter=50, n=10, dim=2, minx=-100, maxx=100)
    print(f"Melhor solução encontrada: {melhor_solucao}")

