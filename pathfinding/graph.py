"""Graph module to represent a graph and perform searches on it:
Path finding algorithms implemented:
- DFS
- BFS
- Djikstra
- A*
"""

import os
import json
import math
import heapq
from collections import deque
from time import perf_counter_ns
from typing import List, Dict, Set, Optional, Any, Generator
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass

GRAPH_FOLDER = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(GRAPH_FOLDER, "graph.json")


class GraphType(Enum):
    DIRECTED = 1
    UNDIRECTED = 2


class Vertex:
    def __init__(self, name: str, index: int = 0):
        self._name: str = name
        self._weight: int = 0
        self._index: int = index
        self._predecessor: Optional[Vertex] = None
        self._distance: int = math.inf

    @property
    def distance(self) -> int:
        return self._distance

    @distance.setter
    def distance(self, distance: int) -> None:
        self._distance = distance

    @property
    def predecessor(self) -> Optional[Any]:
        return self._predecessor

    @predecessor.setter
    def predecessor(self, predecessor: Optional[Any]) -> None:
        self._predecessor = predecessor

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> int:
        return self._weight

    @weight.setter
    def weight(self, weight: int) -> None:
        self._weight = weight

    def __eq__(self, other):
        return self._index == other.index

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


class QueueItem:
    def __init__(self, vertex: Vertex) -> None:
        self.vertex = vertex
        self.distance = vertex.distance

    def __lt__(self, other):
        return self.distance < other.distance


class PriorityQueue:
    def __init__(self, adjlist: List[Vertex] = []) -> None:
        self._queue = []
        if adjlist:
            for vertex in adjlist:
                self._queue.append(QueueItem(vertex))
        heapq.heapify(self._queue)

    def extract_min(self) -> Vertex:
        min_item = heapq.heappop(self._queue)
        return min_item.vertex

    def update(self, vertex: Vertex) -> None:
        for item in self._queue:
            if item.vertex.name == vertex.name:
                item.distance = vertex.distance
                heapq.heapify(self._queue)
                break

    def is_empty(self) -> bool:
        return len(self._queue) == 0


class Graph:
    def __init__(
        self,
        unique_name: str = uuid4().hex,
        graph_type: GraphType = GraphType.UNDIRECTED,
    ):
        self._adj_list: Dict[str, List[Vertex]] = {}
        self._vertexes: Dict[str, Vertex] = {}
        self._graph_name = unique_name
        self._graph_type = graph_type
        self._last_vertex_index = 0

    @property
    def name(self) -> str:
        return self._graph_name

    @property
    def graph_type(self) -> GraphType:
        return self._graph_type

    @property
    def adjacency_list(self) -> Dict[Vertex, List[Vertex]]:
        return self._adj_list

    def get_vertexes(self) -> Dict[str, Vertex]:
        return self._vertexes

    def get_neighbors(self, vertex: Vertex) -> Generator[Vertex, None, None]:
        neighbors = self._adj_list[vertex.name]
        for neighbor in neighbors:
            current_vertex = self.find_vertex_by_name(neighbor[0])
            current_vertex.weight = neighbor[1]
            yield current_vertex

    def find_vertex_by_name(self, name: str) -> Optional[Vertex]:
        if name in self._vertexes:
            return self._vertexes[name]
        return None

    def load_from_json(self, name: str) -> None:
        current_index = 0
        with open(name, mode="r", encoding="utf-8") as file:
            data = json.load(file)
            edges = data["graph"]["edges"]
            self._graph_name = data["graph"]["id"]
            for edge in edges:
                vertex1 = Vertex(edge["source"], current_index)
                vertex2 = Vertex(edge["target"], current_index + 1)
                self.add_edge(vertex1, vertex2, edge["weight"])
                current_index = current_index + 2

    def add_edge(self, vertex1: Vertex, vertex2: Vertex, weight: int) -> None:
        # keep adj list using vertexes name as key and a list of vertexes as value
        if vertex1.name not in self._adj_list:
            # each vertex will have a unique index
            vertex1.index = self._last_vertex_index
            self._adj_list[vertex1.name] = []
            self._last_vertex_index = self._last_vertex_index + 1
        if vertex2.name not in self._adj_list:
            vertex2.index = self._last_vertex_index
            self._last_vertex_index = self._last_vertex_index + 1
            self._adj_list[vertex2.name] = []
        # add the vertexes to the adjacency list
        # direct
        self._adj_list[vertex1.name].append((vertex2.name, weight))
        # this is might an expensive operation to keep the minimum weight at the top
        # kind of greedy optimization, select always locally the best option
        self._adj_list[vertex1.name].sort(key=lambda x: x[1])
        if self._graph_type == GraphType.UNDIRECTED:
            # add reverse edge
            self._adj_list[vertex2.name].append((vertex1.name, weight))
            self._adj_list[vertex2.name].sort(key=lambda x: x[1])

        if vertex1.name not in self._vertexes:
            self._vertexes[vertex1.name] = vertex1
        if vertex2.name not in self._vertexes:
            self._vertexes[vertex2.name] = vertex2


@dataclass
class SearchInfo:
    visited: Set[Vertex]
    edge_to: List[Vertex]


def dfs(graph: Graph, vertex: Vertex, info: SearchInfo) -> List[Vertex]:
    info.visited.add(vertex)
    for neighbor in graph.get_neighbors(vertex):
        if neighbor not in info.visited:
            info.edge_to[neighbor.index] = vertex
            dfs(graph=graph, vertex=neighbor, info=info)


def bfs(graph: Graph, vertex: Vertex, info: SearchInfo) -> List[Vertex]:
    queue = deque()
    queue.append(vertex)
    info.visited.add(vertex)
    while len(queue) > 0:
        current_vertex = queue.popleft()
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor not in info.visited:
                info.visited.add(neighbor)
                info.edge_to[neighbor.index] = current_vertex
                queue.append(neighbor)


def dfs_search(
    graph: Graph, start_vertex: Vertex, end_vertex: Vertex
) -> Optional[bool | deque[Vertex]]:
    search_start = perf_counter_ns()
    info = SearchInfo(visited=set(), edge_to=[])
    # Initialize the edge_to list
    # edge list will be useful to reconstruct the path
    for _ in graph.adjacency_list.keys():
        info.edge_to.append(None)
    # Perform the depth first search
    dfs(graph, start_vertex, info)
    # Initialize the path stack
    path = deque()
    # If we've not marked as visited the end vertex, then there is no path
    if end_vertex not in info.visited:
        return False, []
    # Reconstruct the path from the visited vertices
    current_vertex = end_vertex

    while True:
        path.appendleft(current_vertex)
        current_vertex = info.edge_to[current_vertex.index]
        if current_vertex.index == start_vertex.index:
            break

    path.appendleft(start_vertex)
    end_search = perf_counter_ns()
    performance = end_search - search_start
    return True, path, performance


def bfs_search(
    graph: Graph, start_vertex: Vertex, end_vertex: Vertex
) -> Optional[bool | deque[Vertex]]:
    search_start = perf_counter_ns()
    info = SearchInfo(visited=set(), edge_to=[])
    # Initialize the edge_to list
    # edge list will be useful to reconstruct the path
    for _ in graph.adjacency_list.keys():
        info.edge_to.append(None)
    bfs(graph, start_vertex, info)
    if end_vertex not in info.visited:
        return False, []
    # Initialize the path stack
    path = deque()
    # If we've not marked as visited the end vertex, then there is no path
    if end_vertex not in info.visited:
        return False, []
    # Reconstruct the path from the visited vertices
    current_vertex = end_vertex

    while True:
        path.appendleft(current_vertex)
        current_vertex = info.edge_to[current_vertex.index]
        if current_vertex.index == start_vertex.index:
            break

    path.appendleft(start_vertex)
    for vertex in path:
        print(vertex)
    end_search = perf_counter_ns()
    performance = end_search - search_start
    return True, path, performance


def djikstra_search(
    graph: Graph, start_vertex: Vertex, end_vertex: Vertex
) -> Optional[bool | deque[Vertex]]:
    search_start = perf_counter_ns()
    vertex_map = graph.get_vertexes()
    start_vertex = graph.find_vertex_by_name(start_vertex.name)
    for name, vertex in vertex_map.items():
        vertex.distance = math.inf
        vertex.predecessor = None
        if name == start_vertex.name:
            vertex.distance = 0
    queue = PriorityQueue(adjlist=vertex_map.values())
    queue.update(start_vertex)
    working_set = set()
    while not queue.is_empty():
        current_vertex = queue.extract_min()
        working_set.add(current_vertex.name)
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor.distance > current_vertex.distance + neighbor.weight:
                neighbor.distance = current_vertex.distance + neighbor.weight
                neighbor.predecessor = current_vertex
                queue.update(neighbor)

    path = deque()
    end_vertex = graph.find_vertex_by_name(end_vertex.name)
    if end_vertex.name in working_set:
        current_vertex: Vertex = end_vertex
        while current_vertex is not None:
            path.appendleft(current_vertex)
            current_vertex = current_vertex.predecessor
    else:
        return False, [], 0
    end_search = perf_counter_ns()
    performance = end_search - search_start
    return len(path) > 1, path, performance
