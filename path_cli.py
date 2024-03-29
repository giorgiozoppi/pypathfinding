#!/usr/bin/env python

import typer
import rich
from rich.prompt import Prompt
from pathfinding import Graph, dfs_search, bfs_search, djikstra_search, Vertex
app = typer.Typer()
graph = Graph(unique_name="ireland_roads") 
def ask_for_input(graph: Graph)-> tuple[Vertex, Vertex]:
    start = Prompt.ask("Enter the start node")
    end = Prompt.ask("Enter the end node")
    return graph.find_vertex_by_name(start.strip()), graph.find_vertex_by_name(end.strip())
@app.command()
def test(algorithm_name: str, filename: str):
    graph.load_from_json(filename)
    if algorithm_name == "dfs":
        start, end = ask_for_input(graph)  
        success, path, performance = dfs_search(graph, start, end)
        if success:
            csv_path = ''.join([f"{vertex.name} -> " for vertex in path])
            rich.print(f"[green bold]Path found dfs search from {start.name} -> {end.name}[/green bold]: {csv_path}")
            rich.print(f"[green bold]Elapsed Time: {performance} ns[/green bold]")
        elif start and end:
            rich.print(f"[red bold]No path found bfs search from {start.name} -> {end.name}[/red bold]")
        else:
            rich.print(f"[red bold]No path found dfs search[/red bold]")
    elif algorithm_name == "bfs":
        start, end = ask_for_input(graph)  
        success, path, performance = bfs_search(graph, start, end)
        if success:
            csv_path = ''.join([f"{vertex.name} -> " for vertex in path])
            rich.print(f"[green bold]Path found bfs search from {start.name} -> {end.name}[/green bold]: {csv_path}")
            rich.print(f"[green bold]Elapsed Time: {performance} ns[/green bold]")
        elif start and end:
            rich.print(f"[red bold]No path found bfs search from {start.name} -> {end.name}[/red bold]")
        else:
            rich.print(f"[red bold]No path found bfs search[/red bold]")

    elif algorithm_name == "shortest_path":
        start, end = ask_for_input(graph)
        success, path, performance = djikstra_search(graph, start, end)
        if success:
            csv_path = ''.join([f"{vertex.name} -> " for vertex in path])
            rich.print(f"[green bold]Path found shortest path search from {start.name} -> {end.name}[/green bold]: {csv_path}")
            rich.print(f"[green bold]Elapsed Time: {performance} ns[/green bold]")
        elif start and end:
            rich.print(f"[red bold]No path found shortest path search from {start.name} -> {end.name}[/red bold]")
        else:
            rich.print(f"[red bold]No path found shortest path search[/red bold]")
    else:
        rich.print("[red bold]Invalid algorithm name[/red bold]")


if __name__ == "__main__":
    app()