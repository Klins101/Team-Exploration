# graph_builder.py

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, num_nodes):
        self.graph = {}
        self.num_nodes = num_nodes
        self.nodes = [f'Node{i}' for i in range(self.num_nodes)]
        self.build_graph()

    def build_graph(self):
        # Automatically create nodes
        for node in self.nodes:
            self.add_node(node)

        # Automatically create edges
        for node in self.graph:
            num_edges = random.randint(1, self.num_nodes - 1)
            neighbors = random.sample(self.nodes, num_edges)
            for neighbor in neighbors:
                if neighbor != node:
                    self.add_edge(node, neighbor)

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph:
            if node2 not in self.graph[node1]:
                self.graph[node1].append(node2)
            if node1 not in self.graph[node2]:
                self.graph[node2].append(node1)

    def display_graph(self):
        for node, neighbors in self.graph.items():
            print(f"{node}: {neighbors}")

    def adjacency_matrix(self):
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for i, node in enumerate(self.nodes):
            for neighbor in self.graph[node]:
                j = self.nodes.index(neighbor)
                matrix[i][j] = 1
        return matrix

    def laplacian_matrix(self):
        adj_matrix = self.adjacency_matrix()
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        return laplacian

    def plot_graph(self):
        G = nx.Graph()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_weight='bold')
        plt.show()

if __name__ == "__main__":
    # Example usage: Create a graph with 5 nodes
    num_nodes = 5
    g = Graph(num_nodes)
    g.display_graph()

    print("\nAdjacency Matrix:")
    print(g.adjacency_matrix())

    print("\nLaplacian Matrix:")
    print(g.laplacian_matrix())

    print("\nGraph Plot:")
    g.plot_graph()
