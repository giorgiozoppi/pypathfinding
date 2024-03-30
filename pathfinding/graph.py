"""Graph module to represent a graph and perform searches on it:
Path finding algorithms implemented:
- DFS
- BFS
- Djikstra
- A*
Some design considerations:
- The graph is represented as an adjacency list. The adjacency list is a dictionary
  where the key is the vertex name and the value is a list of tuples. Each tuple contains:
   - the name of the neighbor vertex
   - the weight of the edge between the vertex and the neighbor.
  So visiting the neighbors of a vertex it can cost up to O(V) where V is the number of vertexes.
  Since we're in the need to see all neighbors of a vertex, we can't do better than O(V).

- We store also each vertex by name in a dictionary(self._vertexes) to have a quick access to the vertex.
- Once we've the name of the vertex in adjacency list, we can access the vertex by name using the dictionary,
  this will keep the ajacency list compact, easy to manage and we avoid deep copying.
- The graph can be directed or undirected.

Where a priority queue is used (Shortest Patha and A*), we use a min-heap to keep the minimum value at the top.
We've created a PriorityQueue class to manage the queue inside that we've using heapq module and
hepify method to keep the minimum value at the top.
Each item in the queue is a QueueItem object that wraps a vertex and keep the minimum value at the top.
There are two types of QueueItem objects: DijkstraQueueItem and AStarQueueItem and
depending on the algorithm we use one or another.
"""

import os
import json
import math
import heapq
from collections import deque
from time import perf_counter_ns
from typing import List, Dict, Set, Optional, Any, Generator
from typing_extensions import override
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass
from abc import abstractmethod, ABC
from pydantic import BaseModel

# we suppose to have a kind of graphml file transposed to json
# in the same folder as the graph.py file
GRAPH_FOLDER = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(GRAPH_FOLDER, "graph.json")


class GraphType(Enum):
    """A generic graph can be directed or undirected.
    This emum will help to define the type of the graph.
    """

    DIRECTED = 1
    UNDIRECTED = 2


class Vertex:
    """This class models the Vertex of the Graph.
    Each vertex has a name, a weight, an index, a predecessor, a distance, an h value and a g value.
    - h value is the heuristic value for the A* algorithm.
    - weight is the weight of the edge between two vertexes used in Djikstra algorithm.
    - index is the unique identifier of the vertex.
    - predecessor is the previous vertex in the path.
    - distance is the distance from the start vertex.
    """

    def __init__(self, name: str, distance=float("inf"), index: int = 0) -> None:
        self._name: str = name
        self._weight: int = 0
        self._index: int = index
        self._predecessor: Optional[Vertex] = None
        self._distance: float = distance
        self._h_value: float = float("inf")
        self._g_value: float = 0.0

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

    @property
    def hvalue(self) -> float:
        return self._h_value

    @hvalue.setter
    def hvalue(self, hvalue: float) -> None:
        self._h_value = hvalue

    @property
    def gvalue(self) -> float:
        return self._g_value

    @gvalue.setter
    def gvalue(self, gvalue: float) -> None:
        self._g_value = gvalue

    def __eq__(self, other):
        return self._index == other.index

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


class QueueItemType(Enum):
    """QueueItemType is an enumeration to define the type of the queue item.
    We use a priority queue to keep the minimum value at the top. It is a min-heap.
    In Dijsktra algorithm, the minimum value is the distance from the start vertex.
    In A* algorithm, the minimum value is the sum of the distance from the start vertex
    and the heuristic value.
    """

    DIJKSTRA = 1
    ASTAR = 2


class QueueItem:
    """QueueItem is an abstract class that allow to generalize the items on the work queue."""

    def __init__(self, vertex: Vertex) -> None:
        """Constructor for the QueueItem class."""
        self.vertex = vertex
        self.min_index = self.get_min_index(vertex)

    @abstractmethod
    def get_min_index(self, vertex: Vertex) -> float:
        """Return the value to prioritize of the vertex.
            That value in Dijsktra algorithm is the distance from the start vertex.
            In A* algorithm, the minimum value is the sum of the distance from the start vertex
            and the heuristic value.
        Args:
            vertex (Vertex): Vertex

        Returns:
            float: value to prioritize
        """
        pass

    @abstractmethod
    def set_min_index(self, vertex: Vertex) -> None:
        """Set the minimum index of the vertex.

        Args:
            vertex (Vertex): Vertex to use.
        """
        pass

    def __lt__(self, other) -> bool:
        """Operator (<) overriden to compare two QueueItem objects.

        Args:
            other (QueueItem): QueueItem object to compare.

        Returns:
            bool: True if the current object is less than the other object.
        """
        return self.min_index < other.min_index


class DijkstraQueueItem(QueueItem):
    """Generalization of the QueueItem class for the Dijsktra algorithm."""

    def __init__(self, vertex: Vertex) -> None:
        """Constructor for the DijkstraQueueItem class.

        Args:
            vertex (Vertex): Vertex
        """
        super().__init__(vertex)

    @override
    def get_min_index(self, vertex: Vertex) -> float:
        """Override the get_min_index method to return the distance of the vertex.

        Args:
            vertex (Vertex): Vertex

        Returns:
            float: distance of the vertex from source vertex.
        """
        self.distance = vertex.distance
        return self.distance

    @override
    def set_min_index(self, vertex: Vertex) -> None:
        """Override the set_min_index method to set the distance of the vertex from source vertex.

        Args:
            vertex (Vertex): Vertex
        """
        self.distance = vertex.distance
        self.min_index = vertex.distance


class AStarQueueItem:
    """Generalization of the QueueItem class for the A* algorithm."""

    def __init__(self, vertex: Vertex) -> None:
        """Constructor for the AStarQueueItem class."""
        super().__init__(vertex)
        self.fvalue: float = 0
        self.vertex = vertex

    @override
    def get_min_index(self, vertex: Vertex) -> float:
        """Override the get_min_index method to return the f value of the vertex.

        Args:
            vertex (Vertex): Vertex

        Returns:
            float: heuristic value of the vertex.
        """
        self.fvalue = vertex.gvalue + vertex.hvalue
        return self.fvalue

    @override
    def set_min_index(self, vertex: Vertex) -> None:
        """Set the f value of the vertex.

        Args:
            vertex (Vertex): heurisitc value of the vertex.
        """
        self.fvalue = vertex.gvalue + vertex.hvalue
        self.min_index = self.fvalue


def make_queue_item(
    vertex: Vertex, kind: QueueItemType = QueueItemType.DIJKSTRA
) -> QueueItem:
    """Make queue item based on the kind of the queue item.
       QueueItem is an abstract class that allow to generalize the items on the work queue.
       We can think a QueueItem as wrapper around the vertex to keep the minimum value at the top.
    Args:
        vertex (Vertex): Vertex
        kind (QueueItemType): Algortihm type to adapt the queue item.

    Returns:
        QueueItem: A queue item based on the kind of the queue item.
    """
    if kind == QueueItemType.DIJKSTRA:
        return DijkstraQueueItem(vertex)
    elif kind == QueueItemType.ASTAR:
        return AStarQueueItem(vertex)


class PriorityQueue:
    """Priority Queue class the min-heap."""

    def __init__(self, algorithm: QueueItemType, adjlist: List[Vertex] = []) -> None:
        """Constructor
           Here we've two cases:
           - We can create an empty queue.
           - We can fill up the queue with a list of all vetrexes in the graph.
        Args:
            adjlist (List[Vertex], optional): List of all vertexes. Defaults to [].
        """
        self._queue: List[QueueItem] = []
        self._algorithm = algorithm
        if adjlist:
            for vertex in adjlist:
                self._queue.append(make_queue_item(vertex, kind=self._algorithm))
        # enforce the min-heap property
        heapq.heapify(self._queue)

    def insert(self, vertex: Vertex) -> None:
        """Insert a vertex in the queue.

        Args:
            vertex (Vertex): Vertex
        """
        item = make_queue_item(vertex, kind=self._algorithm)
        heapq.heappush(self._queue, item)

    def extract_min(self) -> Vertex:
        """Extract the Vertex with minimum value from the queue.

        Returns:
            Vertex: Vertex with minimum value.
        """
        min_item = heapq.heappop(self._queue)
        vertex = min_item.vertex
        return vertex

    def update(self, vertex: Vertex) -> None:
        """Add a vertex to the queue and restore the min-heap property.

        Args:
            vertex (Vertex): Vertex to add.
        """
        for item in self._queue:
            if item.vertex.name == vertex.name:
                item.set_min_index(vertex)
                break
        heapq.heapify(self._queue)

    def is_empty(self) -> bool:
        """Return True if the queue is empty.

        Returns:
            bool: True if the queue is empty.
        """
        return len(self._queue) == 0


class Graph:
    """Class to represent a graph.
    A graph can be directed or undirected and it has a unique name.
    During Vertex insertion, we assign a unique index to each vertex.
    """

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
        self.heuristic_table = None
        self._vertex_names: List[str] = []

    @property
    def name(self) -> str:
        """Return the name of the graph.

        Returns:
            str: Name of the graph.
        """
        return self._graph_name

    @property
    def graph_type(self) -> GraphType:
        """Return the type of the graph. It can be directed or undirected.

        Returns:
            GraphType: Type of the graph.
        """
        return self._graph_type

    @property
    def adjacency_list(self) -> Dict[str, List[Vertex]]:
        """Return the adjacency list of the graph.

        Returns:
            Dict[str, List[Vertex]]: The adjacency list of the graph. Each vertex is indexed by name.
        """
        return self._adj_list

    def get_vertexes(self) -> Dict[str, Vertex]:
        """Return all vertexes in the graph indexed by name.

        Returns:
            Dict[str, Vertex]: All vertexes in the graph indexed by name.
        """
        return self._vertexes

    def get_neighbors(self, vertex: Vertex) -> Generator[Vertex, None, None]:
        """Return all neighbors of a vertex.

        Args:
            vertex (Vertex): Vertex
        Yields:
            Generator[Vertex, None, None]: Generator of all neighbors of the vertex.
        """
        neighbors = self._adj_list[vertex.name]
        for neighbor in neighbors:
            current_vertex = self.find_vertex_by_name(neighbor[0])
            current_vertex.weight = neighbor[1]
            yield current_vertex

    def create_heuristic_table(self, destination: Vertex) -> None:
        """Create a heuristic table based on the source names and the destination vertex.

        Args:
            source_names (List[str]): List of source names.
            destination (Vertex): Destination vertex.
        """
        current_names = list(
            filter(lambda x: x != destination.name, self._vertex_names)
        )
        self.heuristic_table = HTable.create(self, current_names, destination)

    def find_vertex_by_name(self, name: str) -> Optional[Vertex]:
        """Find a vertex by name.

        Args:
            name (str): name of the vertex.

        Returns:
            Optional[Vertex]: Vertex if found, None otherwise.`
        """
        if name in self._vertexes:
            return self._vertexes[name]
        return None

    def load_from_json(self, name: str) -> None:
        """Load from a json file the graph. The semantich is the same as the graphml file.

        Args:
            name (str): Path to the json file.
        """
        current_index = 0
        with open(name, mode="r", encoding="utf-8") as file:
            data = json.load(file)
            edges = data["graph"]["edges"]
            self._graph_name = data["graph"]["id"]
            vertexes = data["graph"]["nodes"]
            for vertex in vertexes:
                self._vertex_names.append(vertex["id"])
            for edge in edges:
                vertex1 = Vertex(edge["source"], current_index)
                vertex2 = Vertex(edge["target"], current_index + 1)
                self.add_edge(vertex1, vertex2, edge["weight"])
                current_index = current_index + 2

    def load_heuristic_table(self, name: str) -> None:
        """Load from a json file the heuristic table.

        Args:
            name (str): Path to the json file.
        """
        with open(name, mode="r", encoding="utf-8") as file:
            data = json.load(file)
            self.heuristic_table = HTable(**data)

    def add_edge(self, vertex1: Vertex, vertex2: Vertex, weight: int) -> None:
        """It adds an edge between two vertexes with a weight.
           If the vertexes are not in the adjacency list, they will be added.
           If the graph is undirected, it will add the reverse edge.
           Also it keeps sorted the adjacency list by weight.
           This might be useful where we don't want to use a priority queue (BFS,DFS).
        Args:
            vertex1 (Vertex): Source vertex.
            vertex2 (Vertex): Destination vertex.
            weight (int): Weight of the edge.
        """
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
    """Context to pass thru recursive calls of the search algorithms."""

    visited: Set[Vertex]
    edge_to: List[Vertex]


class HTable(BaseModel):
    """Heuristic table to store the heuristic values."""

    table: Dict[str, Dict[str, float]]

    @classmethod
    def create(cls, g: Graph, source_names: List[str], destination: Vertex):
        table = {}
        for name in sorted(source_names):
            source = g.find_vertex_by_name(name)
            found, path, _ = djikstra_search(g, source, destination)
            if found:
                print(f"Path found from {name} -> {destination.name}")
                table_distance = path[-1].distance
                table[name] = {destination.name: table_distance}
            else:
                print(f"No path found from {name} -> {destination.name}")
                table[name] = {destination.name: float("inf")}

        return cls(table=table)

    def save(self, name: str) -> None:
        json_dict = json.loads(self.model_dump_json())
        with open(name, mode="w", encoding="utf-8") as file:
            json.dump(json_dict, file)


def dfs(graph: Graph, vertex: Vertex, info: SearchInfo) -> List[Vertex]:
    """Define the depth first search algorithm.

    Args:
        graph (Graph): Graph
        vertex (Vertex): Source vertex
        info (SearchInfo): Context to pass thru recursive calls of the search algorithms.

    Returns:
        List[Vertex]: List of vertexes in the path.
    """
    info.visited.add(vertex)
    for neighbor in graph.get_neighbors(vertex):
        if neighbor not in info.visited:
            info.edge_to[neighbor.index] = vertex
            dfs(graph=graph, vertex=neighbor, info=info)


def bfs(graph: Graph, vertex: Vertex, info: SearchInfo) -> List[Vertex]:
    """Define the breadth first search algorithm.

    Args:
        graph (Graph): Graph
        vertex (Vertex): Source vertex
        info (SearchInfo): Context to keep info about visited vertexes.

    Returns:
        List[Vertex]: List of vertexes in the path.
    """
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
) -> Optional[bool | deque[Vertex] | float]:
    """Perform a depth first search on the graph.

    Args:
        graph (Graph): Graph
        start_vertex (Vertex): Source vertex
        end_vertex (Vertex): Destination vertex

    Returns:
        Optional[bool | deque[Vertex] | float]: Path found, path from source to destination, execution time.
    """
    if start_vertex is None or end_vertex is None:
        return False, [], 0
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
) -> Optional[bool | deque[Vertex] | float]:
    """Perform a breadth first search on the graph.

    Args:
        graph (Graph): Graph
        start_vertex (Vertex): Source vertex
        end_vertex (Vertex): Destination vertex

    Returns:
        Optional[bool | deque[Vertex]| float]: Path found, path from source to destination, execution time.
    """
    if start_vertex is None or end_vertex is None:
        return False, [], 0
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
    graph: Graph, start_vertex: Vertex, end_vertex: Vertex, dump: bool = False
) -> Optional[bool | deque[Vertex] | float]:
    """Perform Djikstra Shortest Path algorithm on the graph.

    Args:
        graph (Graph): Path
        start_vertex (Vertex): Source vertex
        end_vertex (Vertex): Destination vertex

    Returns:
        Optional[bool | deque[Vertex]| float]: Path found, path from source to destination, execution time.
    """

    if start_vertex is None or end_vertex is None:
        return False, [], 0
    search_start = perf_counter_ns()
    vertex_map = graph.get_vertexes()
    start_vertex = graph.find_vertex_by_name(start_vertex.name)
    for name, vertex in vertex_map.items():
        vertex.distance = math.inf
        vertex.predecessor = None
        if name == start_vertex.name:
            vertex.distance = 0
    queue = PriorityQueue(adjlist=vertex_map.values(), algorithm=QueueItemType.DIJKSTRA)
    queue.update(start_vertex)
    vertex_visited = set()
    while not queue.is_empty():
        current_vertex = queue.extract_min()
        vertex_visited.add(current_vertex.name)
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor.distance > current_vertex.distance + neighbor.weight:
                neighbor.distance = current_vertex.distance + neighbor.weight
                neighbor.predecessor = current_vertex
                queue.update(neighbor)

    path = deque()
    end_vertex = graph.find_vertex_by_name(end_vertex.name)
    if end_vertex.name in vertex_visited:
        current_vertex: Vertex = end_vertex
        while current_vertex is not None:
            path.appendleft(current_vertex)
            current_vertex = current_vertex.predecessor
    else:
        return False, [], 0
    end_search = perf_counter_ns()
    performance = end_search - search_start
    return len(path) > 1, path, performance


def calculate_h_value(start_vertex: Vertex, end_vertex: Vertex) -> float:
    return 0


def a_star_search(
    graph: Graph, start_vertex: Vertex, end_vertex: Vertex, 
    heristic_file: str
) -> Optional[bool | deque[Vertex]]:
    if start_vertex is None or end_vertex is None:
        return False, [], 0
    search_start = perf_counter_ns()
    start_vertex = graph.find_vertex_by_name(start_vertex.name)
    end_vertex = graph.find_vertex_by_name(end_vertex.name)
    start_vertex.distance = 0
    start_vertex.hvalue = calculate_h_value(start_vertex, end_vertex)
    start_vertex.gvalue = 0
    node_visited = set()
    queue = PriorityQueue()
    queue.insert(start_vertex)
    path = deque()
    while not queue.is_empty():
        current_vertex = queue.extract_min()
        if current_vertex.name == end_vertex.name:
            # reconstruct the path
            path.appendleft(current_vertex)
            break
        else:
            pass
    return len(path) > 1, path, 0
