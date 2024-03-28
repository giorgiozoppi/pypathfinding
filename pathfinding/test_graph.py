import pytest
from pathfinding.graph import Graph, Vertex, GraphType

def test_init():
    graph = Graph(unique_name='test', graph_type=GraphType.UNDIRECTED)
    assert graph.adjacency_list == {}
    assert graph.name == 'test'
    assert graph.graph_type == GraphType.UNDIRECTED

def test_add_edge_first_time():
    graph = Graph()
    v1 = Vertex('Galway')
    v2 = Vertex('Limerick')
    graph.add_edge(vertex1=v1, vertex2=v2, weight=112)
    assert v1 in graph.adjacency_list
    assert v2 in graph.adjacency_list[v1]
    assert v2.weight == 112

def test_add_edge_existing_vertex():
    graph = Graph()
    v1 = Vertex('v1')
    v2 = Vertex('v2')
    v3 = Vertex('v3')
    graph.add_edge(v1, v2, 5)
    graph.add_edge(v1, v3, 3)
    assert v1 in graph._adj_list
    assert v2 in graph._adj_list[v1]
    assert v3 in graph._adj_list[v1]
    assert v2.weight == 5
    assert v3.weight == 3

def test_get_neighbors():
    graph = Graph()
    v1 = Vertex('A')
    v2 = Vertex('B')
    v3 = Vertex('C')
    graph.add_edge(v1, v2, 1)
    graph.add_edge(v1, v3, 2)
    neighbors = graph.get_neighbors(v1)
    assert v2 in neighbors
    assert v3 in neighbors

def test_dfs_search():
    graph = Graph()
    v1 = Vertex('A')
    v2 = Vertex('B')
    v3 = Vertex('C')
    v4 = Vertex('D')
    v5 = Vertex('E')
    graph.add_edge(v1, v2, 1)
    graph.add_edge(v1, v3, 2)
    graph.add_edge(v2, v4, 3)
    graph.add_edge(v3, v5, 4)
    path = graph.dfs_search(v1, v5)
    assert path == [v1, v3, v5]