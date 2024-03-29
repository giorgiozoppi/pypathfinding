from pathfinding.graph import (
    Graph,
    Vertex,
    GraphType,
    GRAPH_FILE,
    dfs_search,
    bfs_search,
    djikstra_search,
)


def test_init():
    graph = Graph(unique_name="ireland_roads", graph_type=GraphType.DIRECTED)
    assert graph.adjacency_list == {}
    assert graph.name == "ireland_roads"
    assert graph.graph_type == GraphType.DIRECTED


def test_add_edge_first_time():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    graph.add_edge(Dublin, Galway, 187)
    assert "Dublin" in graph.adjacency_list.keys()
    assert Galway in graph.adjacency_list["Dublin"]
    assert Galway.weight == 187


def test_add_edge_existing_vertex():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    graph.add_edge(Dublin, Galway, 187)
    graph.add_edge(Dublin, Westport, 222)
    assert "Dublin" in graph.adjacency_list
    assert Galway in graph.adjacency_list["Dublin"]
    assert Westport in graph.adjacency_list["Dublin"]
    assert Galway.weight == 187
    assert Westport.weight == 222


def test_load_from_json():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    duplicate = set()
    for key, _ in graph.adjacency_list.items():
        if key in duplicate:
            assert False
        duplicate.add(key)
    assert len(graph.adjacency_list) > 0


def test_get_neighbors():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    graph.add_edge(Dublin, Galway, 187)
    graph.add_edge(Dublin, Westport, 222)
    neighbors = [x for x in graph.get_neighbors(Dublin)]
    assert Galway in neighbors
    assert Westport in neighbors


def test_find_vertex_by_name():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    vertex = graph.find_vertex_by_name("Galway")
    assert vertex.name == "Galway"


def test_not_found_vertex_by_name():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    vertex = graph.find_vertex_by_name("London")
    assert vertex is None


def test_dfs_fail_search():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    ennis = Vertex("Ennis")
    dooling = Vertex("Dooling")
    graph.add_edge(ennis, dooling, 30)
    source = graph.find_vertex_by_name("Ennis")
    destination = graph.find_vertex_by_name("Tipperary")
    state, _ = dfs_search(graph, source, destination)
    assert state is False


def test_bfs_fail_search():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    ennis = Vertex("Ennis")
    dooling = Vertex("Dooling")
    graph.add_edge(ennis, dooling, 30)
    source = graph.find_vertex_by_name("Ennis")
    destination = graph.find_vertex_by_name("Tipperary")
    state, _ = bfs_search(graph, source, destination)
    assert state is False


def test_dfs_search():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    Sligo = Vertex("Sligo")
    # DFS does the best local choice. It will go to Galway first and then Sligo
    # in our case we've optimized the graph order neighbors to be the less distance.
    graph.add_edge(Dublin, Westport, 222)
    graph.add_edge(Dublin, Galway, 100)
    graph.add_edge(Westport, Sligo, 300)
    graph.add_edge(Sligo, Galway, 500)

    state, path, search_time = dfs_search(graph, Dublin, Sligo)
    assert state is True
    assert path[0].name == Dublin.name
    assert path[1].name == Galway.name
    assert path[2].name == Sligo.name
    assert search_time > 0


def test_bfs_search():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    Sligo = Vertex("Sligo")
    graph.add_edge(Dublin, Westport, 50)
    graph.add_edge(Dublin, Galway, 100)
    graph.add_edge(Westport, Sligo, 300)
    graph.add_edge(Sligo, Galway, 500)

    state, path, search_time = bfs_search(graph, Dublin, Sligo)
    assert state is True
    assert path[0].name == Dublin.name
    assert path[1].name == Westport.name
    assert path[2].name == Sligo.name
    assert search_time > 0


def test_djikstra_search():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    source = graph.find_vertex_by_name("Tipperary")
    destination = graph.find_vertex_by_name("Sligo")
    state, path, search_time = djikstra_search(graph, source, destination)
    assert state is True
    assert path[0].name == "Tipperary"
    assert path[1].name == "Limerick"
    assert path[2].name == "Galway"
    assert path[3].name == "Castlebar"
    assert path[4].name == "Sligo"
    assert search_time > 0


def test_djikstra_fail_search():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    ennis = Vertex("Ennis")
    dooling = Vertex("Dooling")
    graph.add_edge(ennis, dooling, 30)
    source = graph.find_vertex_by_name("Ennis")
    destination = graph.find_vertex_by_name("Tipperary")
    state, _, _ = djikstra_search(graph, source, destination)
    assert state is False
