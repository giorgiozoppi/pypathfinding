from pathfinding.graph import Graph, Vertex, GraphType

def test_init():
    graph = Graph(unique_name="ireland_roads", graph_type=GraphType.DIRECTED)
    assert graph.adjacency_list == {}
    assert graph.name == "ireland_roads"
    assert graph.graph_type == GraphType.DIRECTED
"""
def test_add_edge_first_time():
    graph = Graph()
    Dublin = Vertex('Dublin')
    Galway = Vertex('Galway')
    graph.add_edge(Dublin, Galway,187)
    assert Dublin in graph.adjacency_list.keys()
    assert Galway in graph.adjacency_list[Dublin]
    assert Galway.weight == 187

def test_add_edge_existing_vertex():
    graph = Graph()
    Dublin = Vertex('Dublin')
    Galway = Vertex('Galway')
    Westport = Vertex('Westport')
    graph.add_edge(Dublin, Galway, 187)
    graph.add_edge(Dublin, Westport, 222)
    assert Dublin in graph.adjacency_list
    assert Galway in graph.adjacency_list[Dublin]
    assert Westport in graph.adjacency_list[Dublin]
    assert Galway.weight == 187
    assert Westport.weight == 222
"""