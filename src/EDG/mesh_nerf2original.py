import numpy as np
import open3d as o3d
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name")
parser.add_argument("--mesh_name")
args = parser.parse_args()

mesh_name = args.mesh_name
data_name = args.data_name

mesh_name = f"mesh_dg_{mesh_name}.obj"
mesh_name_new = f"mesh_dg_{mesh_name}_original_size.obj"
mesh = o3d.io.read_triangle_mesh(f"../nerfstudio/mesh_dg/{mesh_name}")

rotation = torch.eye(4)
rotation[1:3, 1:3] = torch.Tensor([[0, -1], [1, 0]])

transform_matrix_, camera_to_world, scale_factor = torch.load(f"../nerfstudio/camera_info/{data_name}_train.pt")
transform_matrix = torch.eye(4)
transform_matrix[:3] = transform_matrix_[:3].cpu()

mesh.scale(1.0 / scale_factor, center = [0,0,0])
mesh.transform(torch.linalg.inv(transform_matrix))
mesh.transform(torch.linalg.inv(rotation))
# mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh(f"../mesh_dg/{mesh_name_new}", mesh)
o3d.io.write_triangle_mesh(f"../../rsync/{mesh_name_new}", mesh)