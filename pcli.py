#!/usr/bin/env python

import typer
import rich
from rich.prompt import Prompt
from pathfinding import Graph, dfs_search, bfs_search, djikstra_search, Vertex, a_star_search
app = typer.Typer()
graph = Graph(unique_name="ireland_roads") 
def ask_for_input(graph: Graph)-> tuple[Vertex, Vertex]:
    start = Prompt.ask("Enter the start node")
    end = Prompt.ask("Enter the end node")
    return graph.find_vertex_by_name(start.strip()), graph.find_vertex_by_name(end.strip())
@app.command()
def test(command: str, filename: str, heuristic_file: str = None):
    graph.load_from_json(filename)

    # create once for all the heuristic table for the graph to use A* algorithm.
    # an heuristic table is a dictionary that contains the heuristic value for each vertex in the graph.
    # In math a table can be seen like a function.
    if command == "heuristic_table":
        graph.create_heuristic_table()
        graph.heuristic_table.save("heuristic_table.json")
        rich.print("[green bold]Heuristic table created[/green bold]")
        return
    elif command == "dfs":
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
    elif command == "bfs":
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

    elif command == "shortest_path":
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
    elif command == "a_star":
        start, end = ask_for_input(graph) 
        graph.load_heuristic_table('heuristic_table.json')
        success, path, performance = a_star_search(graph, start, end, heuristic_file)
        if success:
            csv_path = ''.join([f"{vertex.name} -> " for vertex in path])
            rich.print(f"[green bold]Path A* search from {start.name} -> {end.name}[/green bold]: {csv_path}")
            rich.print(f"[green bold]Elapsed Time: {performance} ns[/green bold]")
        elif start and end:
            rich.print(f"[red bold]No path found shortest path search from {start.name} -> {end.name}[/red bold]")
        else:
            rich.print(f"[red bold]No path found shortest path search[/red bold]")
    else:
        rich.print("[red bold]Invalid command name[/red bold]")
    

if __name__ == "__main__":
    app()