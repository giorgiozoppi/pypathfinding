import copy
from typing import List, Dict, Set
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass
class GraphType(Enum):
    DIRECTED = 1
    UNDIRECTED = 2

class Vertex:
    def __init__(self, name: str, index: int = 0):
        self._name: str = name
        self._weight: int = 0
        self._index: int = index
    @property   
    def index(self)->int:
        return self._index
    @index.setter
    def index(self, index: int)->None:
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

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)
    def __eq__(self, other):
        return self.name == other.name and self.weight == other.weight


class Graph:
    def __init__(self, unique_name: str = uuid4().hex, graph_type: GraphType = GraphType.UNDIRECTED):
        self._adj_list: Dict[Vertex, Set[Vertex]] = {}
        self._graph_name = unique_name
        self._graph_type = graph_type
    @property
    def name(self) -> str:
        return self._graph_name
    @property
    def graph_type(self) -> GraphType:
        return self._graph_type
    @property
    def adjacency_list(self) -> Dict[Vertex, List[Vertex]]:
        return self._adj_list
    
    def get_neighbors(self, vertex: Vertex) -> Set[Vertex]:
        return self._adj_list[vertex]
    
    def add_edge(self, vertex1: Vertex, vertex2: Vertex, weight: int) -> None:
        destination_vertex = vertex2
        destination_vertex.weight = weight
        if vertex1 not in self._adj_list:
            self._adj_list[vertex1] = set()
        if vertex2 not in self._adj_list:
            self._adj_list[vertex2] = set()
        self._adj_list[vertex1].add(destination_vertex)
        if self._graph_type == GraphType.UNDIRECTED:
            source_vertex = vertex1
            source_vertex.weight = weight
            self._adj_list[vertex2].add(source_vertex)
 
@dataclass
class SearchInfo:
    visited: Set[Vertex]
    edge_to: List[Vertex]
    
def dfs(graph: Graph, vertex: Vertex, end_vertex: Vertex, info: SearchInfo) -> List[Vertex]:
    # we reached the end vertex
    info.visited.add(vertex)
    
    for neighbor in graph.get_neighbors(vertex):
        if neighbor not in info.visited:
            info.edge_to[neighbor] = vertex
            dfs(graph=graph, vertex=neighbor, end_vertex=end_vertex, info=info)
       
            
        
      
def dfs_search(graph: Graph, start_vertex: Vertex, end_vertex: Vertex) -> List[Vertex]:
    info = SearchInfo(visited=set(), edge_to=[])
    dfs(graph, start_vertex, end_vertex, info)
    print(info.edge_to)

