import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# Given state as a dictionary
state = {"0": 0, "1": 0, "2": 1, "3": 0, "4": 2, "5": 1, "6": 6, "7": 4, "8": 5, "9": 0, "10": 5, "11": 9, "12": 11, "13": 13, "14": 7, "15": 2, "16": 15, "17": 7, "18": 13, "19": 19, "20": 8, "21": 9, "22": 2, "23": 19, "24": 24, "25": 1, "26": 22, "27": 11, "28": 10, "29": 18, "30": 17, "31": 3, "32": 13, "33": 13, "34": 28, "35": 35, "36": 13, "37": 7, "38": 30, "39": 37, "40": 0, "41": 16, "42": 28, "43": 30, "44": 17, "45": 15, "46": 25, "47": 25}



# Convert keys to integers
state = {int(k): int(v) for k, v in state.items()}

# Initialize a dictionary to keep track of the final destination for each key
final_dest = {}

def find_final_dest(key, state):
    visited = set()
    while key not in visited:
        visited.add(key)
        key = state[key]
    return key

# Find the final destination for each key
for key in state.keys():
    final_dest[key] = find_final_dest(key, state)


# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph based on the final destinations
for start, end in final_dest.items():
    G.add_edge(start, end)

# Plot the graph with a different layout
plt.figure(figsize=(12, 12))
pos = nx.circular_layout(G)  # positions for all nodes, you can try other layouts like 'shell', 'circular', etc.
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18, arrows=True)
plt.show(block=True)