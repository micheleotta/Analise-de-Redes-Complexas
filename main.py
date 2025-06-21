# Grupo 6 - Ana Flávia Martins dos Santos, Fabrício Góes Pinterich, Isabella Vanderlinde Berkembrock
# Leonardo Min Woo Chung, Michele Cristina Otta e Phillip Wan Tcha Yan

from collections import defaultdict
from collections import deque
import pandas as pd
import numpy as np
import heapq
from tqdm import tqdm

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

    def tem_aresta(self, u, v): # verifica se a aresta entre os vértices u e v existe no grafo G e retorna True ou False.
        for aresta in self.adj_list[u]:
            if aresta[0] == v:
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

    def centralidade_grau(self, v):
        return  self.grau(v) / (self.ordem - 1)

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
                        sum_distances += 1 / distances[node][0]
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
        for node in self.adj_list.keys():
            degrees[node] = self.centralidade_grau(node)
        
        sorted_top_10 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_top_10[:10]
        print("\nVértices mais influentes por Grau: ")
        for node, value in top_10:
            print(f"\t{node} : {value}")

    def top_10_centralidade_proximidade(self):
        closeness = {}
        for node in self.adj_list.keys():
            closeness[node] = self.centralidade_proximidade(node)

        sorted_top_10 = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_top_10[:10]

        print("\nVértices mais influentes por Proximidade: ")
        for node, value in top_10:
            print(f"\t{node} : {value}")
            
    def centralidade_intermediacao(self, u=None):
        bet = {v: 0.0 for v in self.adj_list}
        vertices = list(self.adj_list.keys())  # lista vertices
        
        for fonte in tqdm(vertices):
            pilha = []                             # vai guardar a ordem em que visitamos os vértices
            pais = {v: [] for v in vertices}       # para cada vértice, quem são seus "pais" no caminho mais curto
            caminhos = {v: 0.0 for v in vertices}  # quantos caminhos mais curtos existem até cada vértice
            caminhos[fonte] = 1.0                  # da fonte para ela mesma, sempre há exatamente 1 caminho
            dist = {v: np.inf for v in vertices}   # distância mais curta conhecida até cada vértice
            dist[fonte] = 0                        # a distância da fonte para ela mesma é zero
            fila = [(0, fonte)]                    # fila de prioridade: (distância, vértice)
            
            # dijkstra modificado
            while fila:
                d, v = heapq.heappop(fila)
                
                if d > dist[v]:
                    continue
                
                pilha.append(v)
                
                for w, peso in self.adj_list[v]:
                    nova_dist = dist[v] + peso
                    
                    if nova_dist < dist[w]:        
                        dist[w] = nova_dist
                        heapq.heappush(fila, (nova_dist, w))
                        caminhos[w] = caminhos[v]  
                        pais[w] = [v]              
                        
                    elif nova_dist == dist[w]:     
                        caminhos[w] += caminhos[v] 
                        pais[w].append(v)          
            
            # brandes    
            dep = {v: 0.0 for v in vertices}
           
            while pilha:
                w = pilha.pop()
                
                for v in pais[w]:
                    if caminhos[w]:
                        fração = (caminhos[v] / caminhos[w]) * (1 + dep[w])
                        dep[v] += fração
                
                if w != fonte:
                    bet[w] += dep[w]
       
        n = self.ordem
        if n > 2:
            escala = (1 if self.direcionado else 2) / ((n - 1) * (n - 2))
            for v in bet:
                bet[v] *= escala

        if u is None:         
            return bet
        if not self.tem_vertice(u):   
            print(f"Vértice '{u}' não existe no grafo")
            return None
        return bet[u]      
    
    def top_10_centralidade_intermediacao(self):
        bet = self.centralidade_intermediacao()          
        sorted_top_10 = sorted(bet.items(),              
                               key=lambda x: x[1],
                               reverse=True)[:10]
        print("\nVértices mais influentes por Intermediação: ")
        for node, value in sorted_top_10:
            print(f"\t{node} : {value}")
    

    def dfs(self, source_node):
      visited = []
      stack = []

      stack.append(source_node)

      while len(stack) > 0:
        element = stack.pop()

        if element not in visited:
          visited.append(element) 

          for adj, _ in sorted(self.adj_list[element], reverse=True):

            if adj not in visited:
              stack.append(adj)

      return visited

    def arvore_geradora_minima(self, x):
        
        if self.direcionado:
            print("O Algoritmo de Kruskal deve ser executado apenas em grafos não-direcionados.")
            return None

        if not self.tem_vertice(x):
            print(f"Vértice '{x}' não existe no grafo.")
            return None

        # Passo 1: Encontrar a componente que possui x e as arestas a serem adicionadas
        arestas_de_agm = set()
        componente_de_x = self.dfs(x)

        for vertice in componente_de_x:
            for vizinho, peso in self.adj_list[vertice]:
                aresta = (min(vertice, vizinho), max(vertice,vizinho), peso)
                arestas_de_agm.add(aresta)

        
        # Passo 2: Começar com uma Árvore Vazia
        AGM = Grafo(False)
        custo_total = 0

        # Passo 3: Ordenar todos os vértices pelo peso
        arestas_de_agm = list(arestas_de_agm)
        arestas_de_agm.sort(key=lambda x: x[2])

        # Passo 4: Pega a aresta menor, checa se forma um ciclo com a árvore vazia. Se um ciclo é formado, inclui a aresta. Senão, descarta
        # Repetir o passo 4 até que  o número de arestas seja igual a (V - 1)
        
        while arestas_de_agm:
            vertice, vizinho, peso = arestas_de_agm.pop(0)
            
            # Faz DFS na AGM a partir de um dos vértices
            visitados = AGM.dfs(vertice)

            if vizinho not in visitados: # Se o vizinho já estiver nos nós visitados
                AGM.adiciona_aresta(vertice, vizinho)
                AGM.adj_list[vertice][-1] = (vizinho, peso)
                AGM.adj_list[vizinho][-1] = (vertice, peso)
                custo_total += peso 

            if AGM.tamanho == AGM.ordem - 1:
                break

        return AGM, custo_total

    def imprime_lista_adjacencias(self):
        for vertice in self.adj_list:
            print(f"{vertice}: ", end="")
            for adjacente in self.adj_list[vertice]:
                print(f"{adjacente}", end=" -> ")
            print()
        print("\n")
        return ""
    
    def componentes_conexas(self):
        visited = set()
        componentes = []

        for vertice in self.adj_list.keys():
            if vertice not in visited:
                componente = self.dfs(vertice)
                componentes.append(componente)
                visited.update(componente)

        return componentes

    def kosaraju(self):
        def dfs(v, visited, estrutura, grafo, modo):
            visited.add(v)
            for vizinho, _ in grafo[v]:
                if vizinho not in visited:
                    dfs(vizinho, visited, estrutura, grafo, modo)
            if modo == 'pilha':
                estrutura.append(v)
            elif modo == 'componente':
                estrutura.add(v)

        # Etapa 1: DFS no grafo original
        visited = set()
        pilha = []
        for v in self.adj_list:
            if v not in visited:
                dfs(v, visited, pilha, self.adj_list, 'pilha')

        # Etapa 2: transpor o grafo
        transposto = {v: [] for v in self.adj_list}
        for u in self.adj_list:
            for v, _ in self.adj_list[u]:
                transposto[v].append((u, 1))

        # Etapa 3: DFS no grafo transposto
        visited.clear()
        componentes = []
        while pilha:
            v = pilha.pop()
            if v not in visited:
                componente = set()
                dfs(v, visited, componente, transposto, 'componente')
                componentes.append(componente)

        return componentes


# ler o dataset
df = pd.read_csv('netflix_amazon_disney_titles.csv')
df = df[['director', 'cast']]
df.dropna(subset=['director', 'cast'], inplace=True) # ignorar entradas vazias

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
    director_list = (df['director'].iloc[id]).split(', ')
    directors = [padronizar_string(director) for director in director_list]
    for id_a, actor in enumerate(actors):
        # adicionar no grafo direcionado(actor, director)
        # actor -> director
        for director in (directors):
            G1.adiciona_aresta(actor, director)
        # adicionar grafo não direcionado relação entre atores
        for i in range(id_a + 1, len(actors)):
            G2.adiciona_aresta(actor, actors[i])

# retornar quantidade vértices e arestas
print('Grafo 1 (direcionado, ator -> diretor)')
print('Ordem =', G1.ordem)
print('Tamanho =', G1.tamanho)

print('\nGrafo 2 (não direcionado, ator -> ator)')
print('Ordem =', G2.ordem)
print('Tamanho =', G2.tamanho)

# Árvore Geradora Mínima do componente x
x = "BETHANY RISHELL"
AGM, custo_total = G2.arvore_geradora_minima(x)
print(f"\nÁrvore Geradora Mínima da componente que contém {x}: \n")
print(AGM.imprime_lista_adjacencias())
print(f"Custo Total da Árvore Geradora Mínima: {custo_total}")

# Quantidade de componentes
componentes_g1 = G1.kosaraju() 
componentes_g2 = G2.componentes_conexas()  

print('Grafo 1 (direcionado, ator -> diretor)')
print('Ordem =', G1.ordem)
print('Tamanho =', G1.tamanho)
print(f"Quantidade de componentes fortemente conexas em G1: {len(componentes_g1)}")
with open("componentes_G1.txt", "w", encoding="utf-8") as f:
    for i, comp in enumerate(componentes_g1, start=1):
        f.write(f"Componente {i} (tamanho {len(comp)}): {sorted(comp)}\n")


print('\nGrafo 2 (não direcionado, ator -> ator)')
print('Ordem =', G2.ordem)
print('Tamanho =', G2.tamanho)
print(f"Quantidade de componentes conexas em G2: {len(componentes_g2)}")
with open("componentes_G2.txt", "w", encoding="utf-8") as f:
    for i, comp in enumerate(componentes_g2, start=1):
        f.write(f"Componente {i} (tamanho {len(comp)}): {sorted(comp)}\n")
