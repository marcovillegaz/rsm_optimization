import numpy as np

# Sample data
strings = ["a", "b", "c"]
lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Create a dictionary to store the meshes
mesh_dict = {}

# Use meshgrid and store meshes in the dictionary
for key, values in zip(strings, lists):
    mesh_dict[key] = np.meshgrid(*values, indexing="ij")

print(mesh_dict)
