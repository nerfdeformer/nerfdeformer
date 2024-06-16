import torch
import numpy as np
import json
import glob
import open3d as o3d

def mesh_nerf2original(mesh_, data_basename, src_time_id):

    from copy import deepcopy
    mesh = deepcopy(mesh_)
    rotation = torch.eye(4)
    rotation[1:3, 1:3] = torch.Tensor([[0, -1], [1, 0]])

    transform_matrix_, camera_to_world, scale_factor = torch.load(f"../nerfstudio/camera_info/{data_basename}_{src_time_id}_train.pt")
    transform_matrix = torch.eye(4)
    transform_matrix[:3] = transform_matrix_[:3].cpu()

    mesh.scale(1.0 / scale_factor, center = [0,0,0])
    mesh.transform(torch.linalg.inv(transform_matrix))
    mesh.transform(torch.linalg.inv(rotation))
    # mesh.compute_vertex_normals()

    return mesh


task = json.load(open("./task.json", "r"))
for task_name in task.keys():

    data_basename = task_name
    for task_i in task[task_name][:1]:
        
        src_time_id = task_i["src_time_id"]


        # file_list = glob.glob(f"mesh_dg/*{data_basename}_{src_time_id}*_.obj")
        # for x in file_list:
        #     print('x', x)
        #     if not "original_size" in x:
        #         mesh = o3d.io.read_triangle_mesh(x)
        #         mesh = mesh_nerf2original(mesh, data_basename, src_time_id)
        #         o3d.io.write_triangle_mesh(x.replace("_.obj", "_original_size_.obj"), mesh)
        #         print(x.replace("_.obj", "_original_size_.obj"))


        file_list = glob.glob(f"mesh_dg/*{data_basename}_{src_time_id}*.obj")
        for x in file_list:
            print('x', x)
            if not "original_size" in x and x[-5] != "_":
                mesh = o3d.io.read_triangle_mesh(x)
                mesh = mesh_nerf2original(mesh, data_basename, src_time_id)
                o3d.io.write_triangle_mesh(x.replace(".obj", "_original_size.obj"), mesh)
                print(x.replace(".obj", "_original_size.obj"))