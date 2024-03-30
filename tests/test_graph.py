import os.path
from pathfinding.graph import (
    Graph,
    Vertex,
    GraphType,
    GRAPH_FILE,
    dfs_search,
    bfs_search,
    djikstra_search,
    PriorityQueue,
    QueueItemType,
    a_star_search,
)

_HEURISTIC_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "heuristic_table.json")
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
    names = [x for x in graph.adjacency_list.keys()]
    assert "Dublin" in names


def test_add_edge_existing_vertex():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    graph.add_edge(Dublin, Galway, 187)
    graph.add_edge(Dublin, Westport, 222)
    names = [x for x in graph.adjacency_list.keys()]
    assert "Dublin" in names
    assert "Galway" in names
    assert "Westport" in names


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


def test_dfs_search_from_file():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    source = graph.find_vertex_by_name("Tipperary")
    destination = graph.find_vertex_by_name("Sligo")
    state, path, search_time = dfs_search(graph, source, destination)
    assert state is True
    assert path[0].name == "Tipperary"
    assert path[1].name == "Limerick"
    assert path[2].name == "Killarney"
    assert path[3].name == "Cork"
    assert path[4].name == "Waterford"
    assert path[5].name == "Wexford"
    assert path[6].name == "Dublin" 
    assert path[7].name == "Dundalk"
    assert path[8].name == "Belfast"
    assert path[9].name == "Sligo"
    
    
    
    assert search_time > 0

    
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


def test_djikstra_search_killarney_sligo():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    source = graph.find_vertex_by_name("Killarney")
    destination = graph.find_vertex_by_name("Sligo")
    state, _, search_time = djikstra_search(graph, source, destination)
    assert state is True
    assert search_time > 0


def test_djikstra_search_castle_bar_sligo():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    source = graph.find_vertex_by_name("Castlebar")
    destination = graph.find_vertex_by_name("Sligo")
    state, _, search_time = djikstra_search(graph, source, destination)
    assert state is True
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


def test_djikstra_search_cork_sligo():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    source = graph.find_vertex_by_name("Cork")
    destination = graph.find_vertex_by_name("Sligo")
    state, _, search_time = djikstra_search(graph, source, destination)
    assert state is True
    assert search_time > 0


def test_priority_queue_insert():
    queue = PriorityQueue(QueueItemType.DIJKSTRA)
    vertex1 = Vertex("A", 0)
    vertex2 = Vertex("B", 1)
    vertex3 = Vertex("C", 2)
    vertex1.distance = 10
    vertex2.distance = 5
    vertex3.distance = 15
    queue.insert(vertex1)
    queue.insert(vertex2)
    queue.insert(vertex3)
    order_names = []
    assert not queue.is_empty()
    while not queue.is_empty():
        vertex_tmp = queue.extract_min()
        order_names.append(vertex_tmp.name)
    assert order_names == ["B", "A", "C"]


def test_priority_queue_extract_min():
    queue = PriorityQueue(QueueItemType.DIJKSTRA)
    vertex1 = Vertex(name="A", distance=10, index=0)
    vertex2 = Vertex(name="B", distance=5, index=1)
    vertex3 = Vertex(name="C", distance=15, index=2)
    queue.insert(vertex1)
    queue.insert(vertex2)
    queue.insert(vertex3)
    min_vertex = queue.extract_min()
    assert min_vertex.name == vertex2.name


def test_priority_queue_update():
    queue = PriorityQueue(QueueItemType.DIJKSTRA)
    vertex1 = Vertex("A", distance=20, index=0)
    vertex2 = Vertex("B", distance=2, index=1)
    vertex3 = Vertex("C", distance=10, index=2)
    queue.insert(vertex1)
    queue.insert(vertex2)
    queue.insert(vertex3)
    vertex2.distance = 30
    queue.update(vertex2)
    min_vertex = queue.extract_min()
    assert min_vertex.name == vertex3.name


def test_priority_queue_is_empty():
    queue = PriorityQueue(QueueItemType.DIJKSTRA)
    assert queue.is_empty()
    vertex = Vertex("A", distance=10, index=0)
    queue.insert(vertex)
    assert not queue.is_empty()


def test_a_star_search():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Sligo = Vertex("Sligo")
    graph.add_edge(Dublin, Galway, 100)
    graph.add_edge(Sligo, Galway, 200)

    state, path, search_time = a_star_search(graph, Dublin, Sligo, _HEURISTIC_FILE)
    assert state is True
    assert path[0].name == Dublin.name
    assert path[1].name == Galway.name
    assert path[2].name == Sligo.name
    assert search_time > 0


def test_a_star_search_no_path():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    Sligo = Vertex("Sligo")
    graph.add_edge(Dublin, Westport, 222)
    graph.add_edge(Dublin, Galway, 100)
    graph.add_edge(Westport, Sligo, 300)

    state, path, search_time = a_star_search(graph, Dublin, Sligo, _HEURISTIC_FILE)
    assert state is False
    assert len(path) == 0
    assert search_time > 0


def test_a_star_search_invalid_vertices():
    graph = Graph()
    Dublin = Vertex("Dublin")
    Galway = Vertex("Galway")
    Westport = Vertex("Westport")
    Sligo = Vertex("Sligo")
    graph.add_edge(Dublin, Westport, 222)
    graph.add_edge(Dublin, Galway, 100)
    graph.add_edge(Westport, Sligo, 300)
    graph.add_edge(Sligo, Galway, 500)

    state, path, _ = a_star_search(graph, None, Sligo, _HEURISTIC_FILE)
    assert state is False
    assert len(path) == 0

    state, path, _ = a_star_search(graph, Dublin, None, _HEURISTIC_FILE)
    assert state is False
    assert len(path) == 0


def test_a_star_search_empty_graph():
    graph = Graph()
    start_vertex = Vertex("Start")
    end_vertex = Vertex("End")

    state, path, _ = a_star_search(
        graph, start_vertex, end_vertex, _HEURISTIC_FILE
    )
    assert state is False
    assert len(path) == 0


def test_astart_search_graph():
    graph = Graph()
    graph.load_from_json(GRAPH_FILE)
    graph.create_heuristic_table()
    source = graph.find_vertex_by_name("Tipperary")
    destination = graph.find_vertex_by_name("Sligo")
    state, path, search_time = a_star_search(graph, source, destination)
    assert state is True
    assert path[0].name == "Tipperary"
    assert path[1].name == "Limerick"
    assert path[2].name == "Galway"
    assert path[3].name == "Castlebar"
    assert path[4].name == "Sligo"
    assert search_time > 0
