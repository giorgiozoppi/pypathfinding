# CA-1 AI Concepts to Implementation
This repository contains the algorithm implementation in python requestes in Task1:
- DFS
- BFS
- Shortest Path
- A* Search.

To test this repository you need poetry. To install poetry on Linux/WSL:
```bash
$ curl -sSL https://install.python-poetry.org | python3 -
```
Once you've poetry installed, just fetch the dependencies:
```bash
$ poetry lock && poetry install
```
Now you are ready to open a virtual environment:
```bash
$ poetry shell
```
Now that you've all set to run DFS on the graph:
```
$ python pcli.py dfs ./pathfinding/graph.json  
```
For running BFS.
```
$ python pcli.py bfs ./pathfinding/graph.json  
```
For running Dijstra algorithm
```
$ python pcli.py shortest_path ./pathfinding/graph.json  
```
For running A*Search.
```
$ python pcli.py a_start ./pathfinding/graph.json  
```



