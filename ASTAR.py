"""
Simple A* implementation for the supplied graph image.

Graph is represented as a directed adjacency list with positive edge weights.
Heuristic values are taken from the numbers shown inside the nodes in the image
(they are treated as admissible estimates to the goal G2).

This script runs A* from start node 'S' to goal 'G2' and prints the path and
the total cost.

Assumptions (inferred from the image):
 - Start node is 'S'.
 - Goal node is 'G2'.
 - Graph edges and weights are transcribed from the diagram.
 - Heuristic values are the small numbers shown inside nodes and G2 has h=0.

If you want to change start/goal or the graph, edit the `graph` and `heuristic` dicts below.
"""

from heapq import heappush, heappop
from typing import Dict, Tuple, List, Optional


def a_star(graph: Dict[str, Dict[str, float]],
		   h: Dict[str, float],
		   start: str,
		   goal: str) -> Tuple[Optional[List[str]], float]:
	"""Run A* on a directed graph.

	graph: adjacency dict mapping node -> {neighbor: cost, ...}
	h: heuristic mapping node -> estimated cost to goal
	start: start node id
	goal: goal node id

	Returns (path, cost). If no path found, returns (None, inf).
	"""
	open_set = []  # priority queue of (f_score, node)
	heappush(open_set, (h.get(start, 0), start))

	came_from: Dict[str, Optional[str]] = {start: None}
	g_score: Dict[str, float] = {start: 0.0}

	closed = set()

	while open_set:
		f_current, current = heappop(open_set)

		if current == goal:
			# reconstruct path
			path = []
			node = current
			while node is not None:
				path.append(node)
				node = came_from.get(node)
			path.reverse()
			return path, g_score[current]

		if current in closed:
			continue
		closed.add(current)

		neighbors = graph.get(current, {})
		for neigh, cost in neighbors.items():
			tentative_g = g_score[current] + cost
			if neigh in g_score and tentative_g >= g_score[neigh]:
				continue

			# This path to neighbor is better
			came_from[neigh] = current
			g_score[neigh] = tentative_g
			f_score = tentative_g + h.get(neigh, 0)
			heappush(open_set, (f_score, neigh))

	# no path found
	return None, float('inf')


def main():
	# Directed graph transcription from the diagram (edges -> weight)
	graph = {
     'S': {'A': 3, 'B': 1, 'C': 5},
     'A': {'E': 7, 'G1': 10},
     'B': {'C': 2, 'F': 2},
     'C': {'G3': 11},
     'D': {'B': 4, 'S': 6, 'G2': 5},
     'E': {'G1': 2},
     'F': {'D': 1},
     'G1': {},
     'G2': {},
     'G3': {'F': 0}
 }
	# Heuristic values (taken from numbers inside node circles in the image)
	heuristic = {
		'S': 8,
		'A': 9,
		'B': 1,
		'C': 3,
		'D': 4,
		'E': 1,
		'F': 5,
		'G1': 0,
		'G2': 0,  # goal
		'G3': 3,
	}

	start = 'S'
	goal = 'G2'

	path, cost = a_star(graph, heuristic, start, goal)

	if path is None:
		print(f"No path found from {start} to {goal}.")
	else:
		print(f"Path from {start} to {goal}: {' -> '.join(path)}")
		print(f"Total cost: {cost}")


if __name__ == '__main__':
	main()

