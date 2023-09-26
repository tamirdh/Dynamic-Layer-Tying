
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
# Read the states and trainable files
file_path_states = 'states-DQN-wiki2.txt'
file_path_trainable = 'trainable-DQN-wiki2.txt'

with open(file_path_states, 'r') as f_states:
    states_data = f_states.readlines()

with open(file_path_trainable, 'r') as f_trainable:
    trainable_data = f_trainable.readlines()

# Parse the states and trainable data
states_data_parsed = [json.loads(state.strip()) for state in states_data]
trainable_data_parsed = [int(trainable.strip()) for trainable in trainable_data]

# Function to save a graph image for a single state using a simpler layout
def save_graph_image_simple(state, trainable_count, step, output_dir):
    # Initialize the graph
    G = nx.DiGraph()

    # Add nodes and edges based on the state mapping
    for target, source in state.items():
        G.add_edge(source, target)

    # Plotting
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_color='black', font_weight='bold', arrows=True)

    # Display the number of trainable layers
    plt.title(f"Number of Trainable Layers: {trainable_count}")

    # Save the image
    image_path = os.path.join(output_dir, f"graph_step_{step}.png")
    plt.savefig(image_path)
    plt.close()

# Function to save a batch of graph images for a range of states using a simpler layout
def save_graph_images_batch_simple(start, end, states_data, trainable_data, output_dir):
    for step in range(start, end):
        state = states_data[step]
        trainable_count = trainable_data[step]
        save_graph_image_simple(state, trainable_count, step, output_dir)

# Create a directory to store the graph images
output_dir = 'graph_images'
os.makedirs(output_dir, exist_ok=True)

# Batch size
batch_size = 10

# Number of batches
num_batches = len(states_data_parsed) // batch_size
if len(states_data_parsed) % batch_size != 0:
    num_batches += 1

# Generate and save the graph images in batches using a simpler layout
for i in tqdm.trange(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(states_data_parsed))
    save_graph_images_batch_simple(start_idx, end_idx, states_data_parsed, trainable_data_parsed, output_dir)
