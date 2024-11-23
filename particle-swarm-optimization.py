#!/usr/bin/env python3

import random
import copy
import sys

class Particula:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        self.posicao = [0.0] * dim
        self.velocidade = [0.0] * dim
        self.melhor_pos_part = [0.0] * dim

        # Loop para calcular a posição e a velocidade aleatória
        # O intervalo para a posição e velocidade é [minx, maxx]
        for i in range(dim):
            self.posicao[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocidade[i] = ((maxx - minx) * self.rnd.random() + minx)

        # Calcula o fitness da partícula
        self.fitness = fitness(self.posicao)  # fitness atual

        # Inicializa a melhor posição e o melhor fitness da partícula
        self.melhor_pos_part = copy.copy(self.posicao)
        self.melhor_fitness_part = self.fitness  # melhor fitness

# Função de otimização por enxame de partículas (PSO)
def pso(fitness, max_iter, n, dim, minx, maxx):
    # Hiperparâmetros
    inercia = 0.729
    fator_cognitivo = 1.49445
    fator_social = 1.49445

    rnd = random.Random(0)

    # Cria n partículas aleatórias
    enxame = [Particula(fitness, dim, minx, maxx, i) for i in range(n)]

    # Calcula a melhor posição e o melhor fitness do enxame
    melhor_pos_enxame = [0.0 for i in range(dim)]
    melhor_fitness_enxame = sys.float_info.max  # melhor do enxame

    # Encontra a melhor partícula do enxame e seu fitness
    for i in range(n):  # Verifica cada partícula
        if enxame[i].fitness < melhor_fitness_enxame:
            melhor_fitness_enxame = enxame[i].fitness
            melhor_pos_enxame = copy.copy(enxame[i].posicao)

    # Loop principal do PSO
    iteracao = 0
    while iteracao < max_iter:
        # A cada 10 iterações
        # Exibe o número da iteração e o melhor valor de fitness até agora
        if iteracao % 10 == 0 and iteracao > 1:
            print(f"Iteração = {iteracao} melhor fitness = {melhor_fitness_enxame:.3f}")

        for i in range(n):  # Processa cada partícula
            # Calcula a nova velocidade da partícula atual
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()

                enxame[i].velocidade[k] = (
                    (inercia * enxame[i].velocidade[k]) +
                    (fator_cognitivo * r1 * (enxame[i].melhor_pos_part[k] - enxame[i].posicao[k])) +
                    (fator_social * r2 * (melhor_pos_enxame[k] - enxame[i].posicao[k]))
                )

                # Se a velocidade[k] não estiver no intervalo [minx, max], então limita o valor
                if enxame[i].velocidade[k] < minx:
                    enxame[i].velocidade[k] = minx
                elif enxame[i].velocidade[k] > maxx:
                    enxame[i].velocidade[k] = maxx

            # Calcula a nova posição usando a nova velocidade
            for k in range(dim):
                enxame[i].posicao[k] += enxame[i].velocidade[k]

            # Calcula o fitness da nova posição
            enxame[i].fitness = fitness(enxame[i].posicao)

            # A nova posição é a melhor para a partícula?
            if enxame[i].fitness < enxame[i].melhor_fitness_part:
                enxame[i].melhor_fitness_part = enxame[i].fitness
                enxame[i].melhor_pos_part = copy.copy(enxame[i].posicao)

            # A nova posição é a melhor do enxame?
            if enxame[i].fitness < melhor_fitness_enxame:
                melhor_fitness_enxame = enxame[i].fitness
                melhor_pos_enxame = copy.copy(enxame[i].posicao)

        iteracao += 1 
    return melhor_pos_enxame
