from collections import defaultdict
import pandas as pd
import numpy as np
import heapq
from tqdm import tqdm
from time import sleep


# Grafo ponderado -> lista de adjacências
class Grafo:

    # construtor
    def __init__(self, tipo):
        self.ordem = 0
        self.tamanho = 0
        self.adj_list = defaultdict(list)
        self.direcionado = tipo

    def __str__(self):
        return self.imprime_lista_adjacencias()

    def adiciona_vertice(self, u): # adiciona um vértice u (rotulado) ao grafo G.
        # não permite a inclusão de vértices repetidos.
        if self.tem_vertice(u):
            print(f"\nVértice {u} já foi adicionado.\n")
            return
        else:
            self.adj_list[u] = []
            self.ordem += 1

    def adiciona_aresta(self, u, v): # cria uma aresta com peso positivo entre os vértices u e v do grafo G.
        # caso algum dos vértices não exista durante a criação da aresta, ele deve ser criado no grafo.
        if not self.tem_vertice(u):
            self.adiciona_vertice(u)
        if not self.tem_vertice(v):
            self.adiciona_vertice(v)

        # caso a aresta já exista, ela deve ser atualizada com o novo peso.
        for i, (id, peso_atual) in enumerate(self.adj_list[u]):
            if id == v:
                novo_peso = peso_atual + 1
                self.adj_list[u][i] = (v, novo_peso)
                if not self.direcionado: # se não direcionado criar para o outro sentido também v -> u
                    for j, (id2, _) in enumerate(self.adj_list[v]):
                        if id2 == u:
                            self.adj_list[v][j] = (u, novo_peso)
                            return
                return

        # se não existir, cria
        self.adj_list[u].append((v, 1))
        if not self.direcionado:
            self.adj_list[v].append((u, 1))
        self.tamanho += 1

    def tem_vertice(self, u): # verifica se o vértice u existe no grafo G e retorna True ou False.
        if u in self.adj_list:
            return True
        return False

    def grau_entrada(self, u): # retorna a quantidade total de arestas que chegam até o vértice u do grafo G.
        if not self.tem_vertice(u):
            print(f"\nVértice {u} não existe. Não foi possível calcular o grau de entrada.\n")
            return None
        grau_entrada = 0
        for vertice in self.adj_list:
            for item in self.adj_list[vertice]:
                if item[0] == u:
                    grau_entrada += 1
        return grau_entrada

    def grau_saida(self, u): # retorna a quantidade total de arestas que saem do vértice u do grafo G
        if not self.tem_vertice(u):
            print(f"\nVértice {u} não existe. Não foi possível calcular o grau de saida.\n")
            return None
        grau_saida = len(self.adj_list[u])
        return grau_saida

    def grau(self, u): # retorna a quantidade total de arestas conectadas (indegree + outdegree) ao vértice u do grafo G
        if not self.tem_vertice(u):
            print(f"\nVértice {u} não existe. Não foi possível calcular seu grau.\n")
            return None
    
        if self.direcionado:
            return self.grau_entrada(u) + self.grau_saida(u)
        else: # se não direcionado
            return len(self.adj_list[u])

    def centralidade_grau(self, v):

        return  self.grau(v) / (self.ordem - 1)

    def get_peso(self, u, v): # retorna qual é o peso da aresta entre os vértices u e v do grafo G, caso exista uma aresta entre eles
        for vertice in self.adj_list[u]:
            if vertice[0] == v:
                return vertice[1]
        return None
    
    #Usado para calcular o caminho mais curto entre
    #um vertice e os outros dentro do grafo
    def dijkstra(self, u):
        distances = { node : [np.inf, None] for node in self.adj_list.keys()}
        #Altera o valor para o vertice inicial


        distances[u][0] = 0
        prio_queue = [(distances[u][0], u)]
        heapq.heapify(prio_queue)
        while len(prio_queue) > 0:
            current_distance, current_node = heapq.heappop(prio_queue)

            if current_distance > distances[current_node][0]:
                continue

            for neighbor, weight in self.adj_list[current_node]:
                new_distance = current_distance + weight

                if new_distance < distances[neighbor][0]:
                    distances[neighbor][0] = new_distance
                    distances[neighbor][1] = current_node
                    heapq.heappush(prio_queue, (new_distance, neighbor))
        
        return distances


    def centralidade_proximidade(self, u):
        if not self.tem_vertice(u):
            print(f"\nVértice {u} não existe. Não foi possível calcular seu grau.\n")
            return None
        
        #Calcular distâncias dos vértices por dijkstra
        distances = self.dijkstra(u)
        sum_distances = 0

        if self.direcionado:
            #Excluir do calculo vértices não alcançáveis
            for node in distances.keys():
                    if distances[node][1] != None and distances[node][0] != np.inf:
                        sum_distances += (1 / distances[node][0])


            #Calculo da centralidade por proximidade no grafo direcionado
            closeness_centrality = sum_distances / (self.ordem-1)
            return closeness_centrality
        
        else:
            #Excluir do calculo vértices não alcançáveis
            for node in distances.keys():
                    if distances[node][1] != None and distances[node][0] != np.inf:
                        sum_distances += distances[node][0]

            #Calculo da centralidade por proximidade no grafo não direcionado
            closeness_centrality = (self.ordem - 1)/(sum_distances)
            if closeness_centrality > 1 or closeness_centrality < 0:
                closeness_centrality = 0
            return closeness_centrality
        
    def top_10_centralidade_grau(self):
        degrees = {}
        for node in tqdm(self.adj_list.keys()):
            degrees[node] = self.centralidade_grau(node)
        
        sorted_top_10 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_top_10[:10]
        print("\nVértices mais influentes por Grau: ")
        for node, value in top_10:
            print(f"\t{node} : {value}")

    def top_10_centralidade_proximidade(self):
        closeness = {}
        for node in tqdm(self.adj_list.keys()):
            closeness[node] = self.centralidade_proximidade(node)
        
        sorted_top_10 = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_top_10[:10]

        print("\nVértices mais influentes por Proximidade: ")
        for node, value in top_10:
            print(f"\t{node} : {value}")


    def imprime_lista_adjacencias(self):
        print("\n⋅˚₊‧ Lista de adjacências ‧₊˚ ⋅")
        for vertice in self.adj_list:
            print(f"{vertice}: ", end="")
            for adjacente in self.adj_list[vertice]:
                print(f"{adjacente}", end=" -> ")
            print()
        print("\n")
        return ""
    

    def componentes_fortemente_conexos(self):
        def dfs_pilha(v, visitado, pilha):
            visitado.add(v)
            for vizinho, _ in self.adj_list[v]:
                if vizinho not in visitado:
                    dfs_pilha(vizinho, visitado, pilha)
            pilha.append(v)

        def dfs_buscar_componente(v, visitado, componente, transposto):
            visitado.add(v)
            componente.add(v)
            for vizinho, _ in transposto[v]:
                if vizinho not in visitado:
                    dfs_buscar_componente(vizinho, visitado, componente, transposto)

        # Etapa 1: Preenche pilha
        visitado = set()
        pilha = []
        for v in self.adj_list:
            if v not in visitado:
                dfs_pilha(v, visitado, pilha)

        # Etapa 2: Transpoe o grafo
        transposto = {v: [] for v in self.adj_list}
        for u in self.adj_list:
            for v, _ in self.adj_list[u]:
                transposto[v].append((u, 1))

        # Etapa 3: Fazer DFS no grafo transposto seguindo a ordem da pilha
        visitado.clear()
        componentes = []
        while pilha:
            v = pilha.pop()
            if v not in visitado:
                componente = set()
                dfs_buscar_componente(v, visitado, componente, transposto)
                componentes.append(componente)

        return componentes
    
    
    def _dfs_busca_iterativa(self, source_node):
        visited = []
        stack = [source_node]

        while stack:
            element = stack.pop()
            if element not in visited:
                visited.append(element)
                for adj, _ in sorted(self.adj_list[element], reverse=True):
                    if adj not in visited:
                        stack.append(adj)
        return visited

    def dfs_iterativa(self):
            visitado = set()
            componentes = []

            for v in self.adj_list:
                if v not in visitado:
                    componente = set(self._dfs_busca_iterativa(v))
                    visitado.update(componente)
                    componentes.append(componente)

            return componentes











# ler o dataset
df = pd.read_csv('netflix_amazon_disney_titles.csv')
df = df[['director', 'cast']]
df.dropna(subset=['director', 'cast'], inplace=True) # ignorar entradas vazias
df.head()

# formato padronizado dos nomes
def padronizar_string(string):
    string = string.upper().strip()
    return string

# construir os grafos
G1 = Grafo(True)  # grafo direcionado
G2 = Grafo(False) # grafo não direcionado
for id, cast in enumerate(df['cast']):
    actors_list = cast.split(', ')
    actors = [padronizar_string(actor) for actor in actors_list]
    for id_a, actor in enumerate(actors):
        # adicionar no grafo direcionado(actor, director)
        # actor -> director
        director_list = (df['director'].iloc[id]).split(', ')
        directors = [padronizar_string(director) for director in director_list]
        for director in (directors):
            G1.adiciona_aresta(actor, director)
        # adicionar grafo não direcionado relação entre atores
        for i in range(id_a + 1, len(actors)):
            G2.adiciona_aresta(actor, actors[i])

print('Grafo 1 (direcionado, ator -> diretor)')
print('Ordem =', G1.ordem)
print('Tamanho =', G1.tamanho)

componentes_g1 = G1.componentes_fortemente_conexos()
componentes_g2 = G2.dfs_iterativa()

with open("componentes_G1.txt", "w", encoding="utf-8") as f:
    for i, componente in enumerate(componentes_g1, start=1):
        f.write(f"Componente {i} (tamanho {len(componente)}): {sorted(componente)}\n")

print(f"Total de componentes fortemente conexos no G1: {len(componentes_g1)}\n")



print('\nGrafo 2 (não direcionado, ator -> ator)')
print('Ordem =', G2.ordem)
print('Tamanho =', G2.tamanho)

with open("componentes_G2.txt", "w", encoding="utf-8") as f:
    for i, componente in enumerate(componentes_g2, start=1):
        f.write(f"Componente {i} (tamanho {len(componente)}): {sorted(componente)}\n")

print(f"Total de componentes conexos em G2: {len(componentes_g2)}")








