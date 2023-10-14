from quantum_models import *
import torch
import torch.distributed as dist
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
dist.init_process_group(backend='gloo', init_method='tcp://localhost:23459', rank=0, world_size=1)

# model = torch.load("/Users/tamirdavidhay/Downloads/models", map_location=torch.device('cpu'))
model = torch.load("/Users/tamirdavidhay/Downloads/gpt-model.pt", map_location=torch.device('cpu'))

# model: DynamicGPT = model.module
# model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-xl")
state_dict = model.transformer.state_dict()
# state = model.state
matching_indices = [0, 2, 4, 5, 9, 10, 36]

def extract_weights(state_dict, layer_indices):
    weights = []
    for idx in layer_indices:
        if idx <= 23:
            weight_key = f"trainable_blocks.{idx}.mlp.c_fc.weight"
        else:
            weight_key = f"non_trainable_blocks.{idx-24}.mlp.c_fc.weight"
        if weight_key in state_dict:
            weights.append(state_dict[weight_key])
    return weights

def compute_similarity_matrix(weight_list):
    n = len(weight_list)
    sim_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight_i = weight_list[i].flatten()
            weight_j = weight_list[j].flatten()
            weight_i = F.normalize(weight_i, p=2, dim=0)
            weight_j = F.normalize(weight_j, p=2, dim=0)
            sim = F.cosine_similarity(weight_i.unsqueeze(0), weight_j.unsqueeze(0))
            sim_matrix[i, j] = sim if i!=j else 1.0
            # Debug: print the values when i == j

    return sim_matrix

# Extract the weights for the specified layers
weight_list = extract_weights(state_dict, matching_indices)

# Compute the similarity matrix
sim_matrix = compute_similarity_matrix(weight_list).numpy()

# Plot the similarity matrix
plt.figure(figsize=(10, 10))
layers = [0, 2, 4, 5, 9, 10, 36]

# Create the heatmap
sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', xticklabels=layers, yticklabels=layers,
            annot_kws={"size": 16})  # Enlarge the font size of annotations

# Enlarge the font size of axis labels and title
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Layer Similarity Matrix", fontsize=18)
plt.xlabel("Untied Layer", fontsize=16)
plt.ylabel("Untied Layer", fontsize=16)

# Reduce the white-space
plt.tight_layout()

# Show the plot
plt.show()