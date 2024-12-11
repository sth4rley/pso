import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
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

        for i in range(dim):
            self.posicao[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocidade[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness_function = fitness_function
        self.fitness = self.fitness_function(self.posicao)
        self.melhor_pos_part = copy.copy(self.posicao)
        self.melhor_fitness_part = self.fitness

# Classe para Visualização
class GraficoPSO:
    def __init__(self, fitness_function, minx, maxx):
        self.fig = plt.figure(figsize=(12, 6))

        # Gráfico 3D
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        x = np.linspace(minx, maxx, 40)
        y = np.linspace(minx, maxx, 40)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[fitness_function([xi, yi]) for xi in x] for yi in y])

        self.surfc = self.ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)
        self.particles_scatter3d = self.ax3d.scatter([], [], [], color="blue", label="Partículas")
        self.best_scatter3d = self.ax3d.scatter([], [], [], color="red", label="Melhor solução")

        self.ax3d.legend()
        self.ax3d.set_xlim(minx, maxx)
        self.ax3d.set_ylim(minx, maxx)
        self.ax3d.set_zlim(Z.min(), Z.max())
        self.ax3d.set_title("PSO: Otimização Animada 3D")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Fitness")
        self.fig.colorbar(self.surfc, ax=self.ax3d, label="Fitness")

        # Gráfico 2D
        self.ax2d = self.fig.add_subplot(122)
        self.contour = self.ax2d.contourf(X, Y, Z, levels=40, cmap="viridis")
        self.particles_scatter2d = self.ax2d.scatter([], [], color="blue", label="Partículas")
        self.best_scatter2d = self.ax2d.scatter([], [], color="red", label="Melhor solução")
        self.ax2d.legend()
        self.ax2d.set_xlim(minx, maxx)
        self.ax2d.set_ylim(minx, maxx)
        self.ax2d.set_title("Projeção 2D do Fitness")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Y")

    def update(self, enxame, melhor_pos_enxame, melhor_fitness_enxame, frame):
        x_data = [p.posicao[0] for p in enxame]
        y_data = [p.posicao[1] for p in enxame]
        z_data = [p.fitness_function(p.posicao) for p in enxame]

        # Atualiza o gráfico 3D
        self.particles_scatter3d._offsets3d = (x_data, y_data, z_data)
        self.best_scatter3d._offsets3d = ([melhor_pos_enxame[0]], [melhor_pos_enxame[1]], [melhor_fitness_enxame])
        self.ax3d.set_title(f"Iteração {frame + 1} - Melhor Fitness = {melhor_fitness_enxame:.6f}")

        # Atualiza o gráfico 2D
        self.particles_scatter2d.set_offsets(np.c_[x_data, y_data])
        self.best_scatter2d.set_offsets([[melhor_pos_enxame[0], melhor_pos_enxame[1]]])

# PSO com visualização
def pso(fitness_function, max_iter, n, dim, minx, maxx):
    # Hiperparâmetros
    inercia = 0.729
    fator_cognitivo = 1.49445
    fator_social = 1.49445

    # Cria n partículas aleatórias
    enxame = [Particula(fitness_function, dim, minx, maxx, i) for i in range(n)]

    # Melhor solução global
    melhor_pos_enxame = [0.0 for _ in range(dim)]
    melhor_fitness_enxame = float('inf')

    # Inicializa visualização
    visualizacao = GraficoPSO(fitness_function, minx, maxx)

    rnd = random.Random(0)

    # Função para atualizar o gráfico em cada frame
    def update(frame):
        nonlocal melhor_fitness_enxame, melhor_pos_enxame

        # Atualiza cada partícula
        for i in range(n):
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

                enxame[i].velocidade[k] = max(minx, min(enxame[i].velocidade[k], maxx))
                enxame[i].posicao[k] += enxame[i].velocidade[k]

            enxame[i].fitness = fitness_function(enxame[i].posicao)

            if enxame[i].fitness < enxame[i].melhor_fitness_part:
                enxame[i].melhor_fitness_part = enxame[i].fitness
                enxame[i].melhor_pos_part = copy.copy(enxame[i].posicao)

            if enxame[i].fitness < melhor_fitness_enxame:
                melhor_fitness_enxame = enxame[i].fitness
                melhor_pos_enxame = copy.copy(enxame[i].posicao)

        # Atualiza o gráfico
        visualizacao.update(enxame, melhor_pos_enxame, melhor_fitness_enxame, frame)

    # Configura a animação
    ani = FuncAnimation(visualizacao.fig, update, frames=max_iter, interval=100, repeat=False)

    plt.show()
    return melhor_pos_enxame

# Função Quadrática (Parabólica)
def fitness_quadratica(pos):
    x, y = pos
    return x**2 + y**2

# Função de Ackley
def fitness_ackley(pos):
    x, y = pos
    return -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20

if __name__ == "__main__":
    melhor_solucao = pso(fitness_quadratica, max_iter=100, n=20, dim=2, minx=-100, maxx=100)
    melhor_solucao = pso(fitness_ackley, max_iter=100, n=20, dim=2, minx=-100, maxx=100)
    print(f"Melhor solução encontrada: {melhor_solucao}")
