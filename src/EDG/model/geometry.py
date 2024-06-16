import torch
import numpy as np
import MVRegC
import open3d as o3d
import os
from orderedset import OrderedSet
import kornia.geometry.conversions as conversions


def ED_warp_old(x, g, R, t, w):
    """ Warp a point cloud using the embeded deformation
    https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
    :param x: point location # [N,3]
    :param g: anchor location # [N,6,3]
    :param R: rotation # [N,6,3,3]
    :param t: translation # [N,6,3]
    :param w: weights # [N,6]
    :return:
    """
    # R.shape: [N,6,3,3] x: [N,1,3] - [N,6,3] -> 
    # print('dd', R.shape, x.shape, g.shape)
    # print('debug', (R * (x[:,None] - g)[:,:,None]).sum(-1).shape, g.shape, t.shape, w.shape)
    # print('debb', x.mean(), g.mean(), t.mean(), w.mean())
    y = ( (R * (x[:,None] - g)[:,:,None]).sum(-1) + g + t ) * w[...,None]
    # y = (R @ (x - g) + g + t ) * w （注意这里有两种g的表示，可以这样，也可以使用原本vertex的所在地方）
    # y = ( R * (x[:,None] - g) + g + t ) * w[...,None]
    y = y.sum(dim=1) # [N,3]
    return y

def calc_average_rotation_matrix(rot_mats, w): # rot_mats: [N,6,3,3], w: [N,6]

    q = conversions.rotation_matrix_to_quaternion(rot_mats) # [N,6,4]
    M = torch.einsum('bri,br,brj->brij', q, w, q) # [N,6,4,4]
    M = M.sum(dim=1) # [N,4,4]
    eigenvalues, eigenvectors = torch.linalg.eigh(M) # [N,4], [N,4,4]
    # max_eigenvalue_index = torch.argmax(eigenvalues, dim = -1)
    # max_eigenvectors = eigenvectors.gather(dim = 1, index = max_eigenvalue_index[:,None,None].repeat(1,1,4)).squeeze(1)
    max_eigenvectors = eigenvectors[:,:,3]
    avg_quaternion = max_eigenvectors / torch.norm(max_eigenvectors, dim = -1, keepdim = True)
    R = conversions.quaternion_to_rotation_matrix(avg_quaternion)
    return R

def ED_warp(x, g, R, t, w):
    """ Warp a point cloud using the embeded deformation
    https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
    :param x: point location # [N,3]
    :param g: anchor location # [N,6,3]
    :param R: rotation # [N,6,3,3]
    :param t: translation # [N,6,3]
    :param w: weights # [N,6]
    :return:
    """
    t = t * w.unsqueeze(-1) # [N,6,3]
    t = t.sum(dim = 1) # [N,3]
    return x + t

    R = calc_average_rotation_matrix(R, w) # [N,3,3]
    # y = (R * (x[:,None] - g)
    # y = ( (R * (x[:,None] - g)[:,:,None]).sum(-1) + g + t ) * w[...,None]
    return y

def average_rotation_matrix(rot_mats, weights):
    """
    Average rotation matrices using quaternion representation.
    
    Parameters:
    - rot_mats: A tensor of shape (N, 3, 3) containing N rotation matrices.
    - weights: A tensor of shape (N,) containing the weights for each rotation matrix.
    
    Returns:
    - avg_rot_mat: The averaged rotation matrix.
    """
    # Ensure that the input is a float tensor for kornia functions
    rot_mats = rot_mats.float()
    weights = weights.float()
    
    # Normalize the weights
    weights = weights / weights.sum()
    
    # Convert rotation matrices to quaternions
    quaternions = conversions.rotation_matrix_to_quaternion(rot_mats)
    
    # Compute the weighted outer product sum of quaternions
    M = torch.zeros(4, 4, device=rot_mats.device, dtype=torch.float32)
    for i in range(quaternions.size(0)):
        q = quaternions[i]
        w = weights[i]
        M += w * torch.outer(q, q)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    max_eigenvalue_index = torch.argmax(eigenvalues)
    avg_quaternion = eigenvectors[:, max_eigenvalue_index]
    avg_quaternion = avg_quaternion / torch.norm(avg_quaternion)
    avg_rot_mat = conversions.quaternion_to_rotation_matrix(avg_quaternion)
    
    return avg_rot_mat



def xyz_2_uv(pcd, intrin):
    '''
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx).astype(int)
    v = (fy * Y / Z + cy).astype(int)
    return np.stack([u,v], -1 )


def depth_2_pc(depth, intrin):
    '''
    :param depth:
    :param intrin: 3x3 mat
    :return:
    '''

    fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    height, width = depth.shape
    u = np.arange(width) * np.ones([height, width])
    v = np.arange(height) * np.ones([width, height])
    v = np.transpose(v)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.stack([X, Y, Z])


def depth_to_mesh(depth_image,
                 mask_image,
                 intrin,
                 depth_scale=1000.,
                 max_triangle_distance=0.04):
    """
    :param depth_image:
    :param mask_image:
    :param intrin:
    :param depth_scale:
    :param max_triangle_distance:
    :return:
    """
    width = depth_image.shape[1]
    height = depth_image.shape[0]


    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    point_image = depth_2_pc(depth_image / depth_scale, intrin)
    point_image = point_image.astype(np.float32)

    vertices, faces, vertex_pixels = MVRegC.depth_to_mesh(point_image, max_triangle_distance)

    return vertices, faces, vertex_pixels, point_image


def compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS ) :

    num_nodes = node_indices.shape[0]
    num_vertices = vertices.shape[0]

    graph_edges              = -np.ones((num_nodes, num_neighbors), dtype=np.int32)
    graph_edges_weights      =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    graph_edges_distances    =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

    visible_vertices = np.ones_like(valid_vertices)
    MVRegC.compute_edges_geodesic( vertices, visible_vertices, faces, node_indices,
                                   graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances,
                                   num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )

    return graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances


def argmin_dis_batch(p, q, K=1): # query each point of p (N) in q (M), dis shape: [N, K]
    print("argmin_dis_batch (pytorch3d)", p.shape, q.shape)
    from pytorch3d import ops
    print('pq', p.shape, q.shape)
    result = ops.knn_points(p[None], q[None], K=K)

    return result[0][0, :, :], result[1][0, :, :] # 

def get_pt_anchor_weight(vertices_id_list, n_neighbor, node_vertex_dis, node_coverage):

    pt_anchor = []
    pt_weight = []
    for v in vertices_id_list:
        dis_and_id = []
        # print('in 1', v, node_vertex_s[:,v].sum())
        for node in range(node_vertex_dis.shape[0]):
            dis = node_vertex_dis[node, v]
            if dis < 0:
                continue
            dis_and_id.append([dis, node])
        dis_and_id = sorted(dis_and_id, key=lambda x: x[0])
        pt_anchor.append([x[1] for x in dis_and_id[:n_neighbor]])
        weight = np.array([x[0] for x in dis_and_id[:n_neighbor]])
        # print('w in 1', weight)
        weight = np.exp(- (weight * weight) / (2 * node_coverage * node_coverage))
        weight = weight / np.sum(weight)
        for j in range(n_neighbor - len(pt_anchor[-1])):
            pt_anchor[-1].append(-1)
            weight = np.append(weight, 0)
        pt_weight.append(weight)

    pt_anchor = np.stack(pt_anchor, axis=0)
    pt_weight = np.stack(pt_weight, axis=0)
    return pt_anchor, pt_weight

def get_pt_anchor_weight_minibatch(n_neighbor, vertex_node_dis, node_coverage):

    vertex_node_dis = torch.from_numpy(vertex_node_dis).cuda()
    # print('su minibatch', vertex_node_dis.sum())
    vertex_node_dis[vertex_node_dis < -1e-8] = 1e10
    
    sorted_dis, indices = torch.sort(vertex_node_dis)
    weight = sorted_dis[:, :n_neighbor]
    # print('w in minibatch', weight)
    weight = torch.exp(- (weight * weight) / (2 * node_coverage * node_coverage))
    anchor = indices[:, :n_neighbor]
    weight = weight / torch.sum(weight, dim=1, keepdim=True)
    del vertex_node_dis, sorted_dis, indices
    return anchor, weight

def get_pt_anchor_weight_batch(n_neighbor, node_vertex_dis, node_coverage, V_fixed_id_list = None):

    pt_anchor = []
    pt_weight = []

    vertex_node_dis = node_vertex_dis.T
    if V_fixed_id_list is not None:
        print('minibatch info', vertex_node_dis.shape, V_fixed_id_list.shape)
        vertex_node_dis = vertex_node_dis[V_fixed_id_list]
        if V_fixed_id_list.shape[0] == 1:
            vertex_node_dis = vertex_node_dis[None]
        print(vertex_node_dis.shape, V_fixed_id_list)
    bs = int(1e6 // vertex_node_dis.shape[1])
    print('bs', bs, vertex_node_dis.shape)
    for i in range(0, vertex_node_dis.shape[0], bs):
        print('anchor batch', i, i + bs)
        with torch.no_grad():
            anchor, weight = get_pt_anchor_weight_minibatch(n_neighbor, vertex_node_dis[i:i+bs], node_coverage)
            pt_anchor.append(anchor.cpu())
            pt_weight.append(weight.cpu())
    pt_anchor = torch.cat(pt_anchor, axis=0)
    pt_weight = torch.cat(pt_weight, axis=0)
    return pt_anchor, pt_weight

def truncate(a):

    b = []
    for x in a:
        bb = []
        for y in x:
            bb.append(float('%.6g' % y))
        b.append(bb)
    return b

def get_extra_triangle(nodes_tri, nodes_tetra_sf, nodes_all, all_is_sf, tetra, middle_save_name):

    map_tetra_sf_to_tri = torch.ones(nodes_tetra_sf.shape[0]).long() * -1
    map_all_to_tetra = torch.ones(len(nodes_all)).long() * -1
    map_all_to_tri   = torch.ones(len(nodes_all)).long() * -1
    min_dis_all, id_list = argmin_dis_batch(nodes_tetra_sf, nodes_tri.cuda().float())
    # print('min_dis_all', min_dis_all.mean(), min_dis_all.max(), min_dis_all.min()) # max is around 1e-6, mean is around 1e-7
    for i,x in enumerate(id_list):
        map_tetra_sf_to_tri[i] = x[0]
    cnt_tetra = nodes_tri.shape[0]
    cnt_tri_sf = 0
    for i in range(nodes_all.shape[0]):
        if all_is_sf[i]:
            map_all_to_tri[i] = map_tetra_sf_to_tri[cnt_tri_sf]
            cnt_tri_sf += 1
        else:
            map_all_to_tetra[i] = cnt_tetra
            cnt_tetra += 1
    extra_triangle = []
    if os.path.exists(f'./{middle_save_name}.pt'):
        print('load extra triangle', middle_save_name)
        extra_triangle = torch.load(f'./{middle_save_name}.pt')
        return extra_triangle
    for ii, x in enumerate(tetra):
        if ii % 1000 == 0:
            print('extra triangle', ii, len(tetra))
        # if ii > 10000:
        #     break
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(j + 1, 4):
                    extra_triangle.append([x[i] - 1, x[j] - 1, x[k] - 1])
                    for l in range(3):
                        if all_is_sf[extra_triangle[-1][l]]:
                            extra_triangle[-1][l] = map_all_to_tri[extra_triangle[-1][l]]
                        else:
                            extra_triangle[-1][l] = map_all_to_tetra[extra_triangle[-1][l]]
    extra_triangle = torch.Tensor(extra_triangle)
    print('save extra triangle', middle_save_name)
    torch.save(extra_triangle, f'./{middle_save_name}.pt')
    extra_triangle = torch.load(f'./{middle_save_name}.pt')
    return extra_triangle

def read_msh(f_name_tri, f_name_tetra_sf, f_name_tetra):

    nodes_all = []
    tetra = []
    print('read_msh', f_name_tri, f_name_tetra_sf, f_name_tetra)
    # print(os.path.exists(f_name_tri), os.path.exists(f_name_tetra_sf), os.path.exists(f_name_tetra))
    # input()
    ss = open(f_name_tetra).readlines()
    mesh_tri = o3d.io.read_triangle_mesh(f_name_tri)
    nodes_tri = torch.from_numpy(np.asarray(mesh_tri.vertices))
    mesh_tetra_sf = o3d.io.read_triangle_mesh(f_name_tetra_sf)
    nodes_tetra_sf = torch.from_numpy(np.asarray(mesh_tetra_sf.vertices))
    for s, i in zip(ss, range(len(ss))):
        # print(s)
        if not s.startswith('$Nodes'):
            continue
        n_nodes_all = int(ss[i + 1].split(' ')[-1])
        id = i + 2
        # print('a', n_nodes_all, n_nodes_tri, n_nodes_tetra, id)
        print('debug 1 ', ss[id -1: id + 2])
        for i in range(id, id + n_nodes_all):
            s = ss[i]
            # if np.random.random() <= 0.001:
            #     print(n_nodes_all, i, [float(x) for x in s.replace('\n', '').split(' ')[1:4]])
            nodes_all.append([float(x) for x in s.replace('\n', '').split(' ')[1:4]])
        id = id + n_nodes_all + 2
        print('debug 2 ', ss[id -1: id + 2])
        n_elements_tetra = int(ss[id].split(' ')[-1])
        id += 1
        for i in range(id, id + n_elements_tetra):
            s = ss[i]
            # if np.random.random() <= 0.001:
            #     print(n_elements_tetra, i, [int(x) for x in s.split(' ')[3:7]])
            tetra.append([int(x) for x in s.split(' ')[3:7]])
        # if s.startswith('$Elements'):
        #     n_tetra = int(ss[ss.index(s) + 1])
        #     for i in range(n_tetra):
        #         s = ss[ss.index(s) + 2 + i]
        #         if s.split()[1] == '4':
        #             tetra.append([int(x) for x in s.split()[-4:]]) # 4 is the code for tetrahedron
    nodes_tetra_sf_truncated = truncate(nodes_tetra_sf)
    nodes_all = torch.Tensor(nodes_all).cuda().float()
    min_dis_all, id_list = argmin_dis_batch(torch.Tensor(nodes_tetra_sf_truncated).cuda().float(), nodes_all)
    is_sf = min_dis_all < 0.1 ** 8
    all_is_sf = torch.zeros(len(nodes_all)).cuda().bool()
    for i in range(len(is_sf)):
        if is_sf[i]:
            print('a', id_list[i,0])
            all_is_sf[id_list[i,0]] = True
    print(all_is_sf.sum())
    print(nodes_all.shape, all_is_sf.shape, nodes_all[0])
    # torch.save([all_is_sf.cpu(), nodes_all.cpu()], '../rsync/temp.pth')
    nodes_tetra_sf = nodes_all[all_is_sf]
    nodes_tetra = nodes_all[~all_is_sf]
    
    extra_triangle = get_extra_triangle(nodes_tri, nodes_tetra_sf, nodes_all, all_is_sf, tetra, os.path.basename(f_name_tetra) + "_")
    return nodes_tri, nodes_tetra, extra_triangle
    # sf_points = o3d.geometry.PointCloud()
    # sf_points.points = o3d.utility.Vector3dVector(nodes_all[all_is_sf].cpu().numpy())
    # sf_points.paint_uniform_color([1, 0, 0])
    # inner_points = o3d.geometry.PointCloud()
    # inner_points.points = o3d.utility.Vector3dVector(nodes_all[~all_is_sf].cpu().numpy())
    # inner_points.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([sf_points, inner_points])

   
    
    
    # torch.save(min_dis_all.cpu(), 'min_dis.pth')
    for i in range(5, 15):
        print(i, (min_dis_all < 0.1 ** 8).sum(), nodes_tetra_sf.shape)
    input('pause')
    return nodes_tri, nodes_tetra, tetra
 

def read_msh_old(f_name): # read .msh file (gmsh format)

    # I need to get position of all nodes, and the connectivity of all tetrahedrons
    nodes_tri = []
    nodes_tetra = []
    tetra = []
    ss = open(f_name).readlines()
    for s, i in zip(ss, range(len(ss))):
        # print(s)
        if not s.startswith('$Nodes'):
            continue
        n_nodes_all = int(ss[i + 1].split(' ')[-1])
        n_nodes_tri = int(ss[i + 2].split(' ')[-1])
        n_nodes_tetra = n_nodes_all - n_nodes_tri
        id = i + 3 + n_nodes_tri
        # print('a', n_nodes_all, n_nodes_tri, n_nodes_tetra, id)
        print('debug 1 ', ss[id -1: id + 2])
        for i in range(id, id + n_nodes_tri):
            s = ss[i]
            nodes_tri.append([float(x) for x in s.split(' ')[:3]])
        id = id + n_nodes_tri + 1 + n_nodes_tetra
        print('debug 2 ', ss[id -1: id + 2])
        
        for i in range(id, id + n_nodes_tetra):
            s = ss[i]
            nodes_tetra.append([float(x) for x in s.split(' ')[:3]])
        id = id + n_nodes_tetra + 2
        print('debug 3 ', ss[id -1: id + 2])
        n_elements_all = int(ss[id].split(' ')[-1])
        n_elements_tri = int(ss[id + 1].split(' ')[-1])
        id = id + 2 + n_elements_tri
        print('debug 4 ', ss[id -1: id + 2])
        n_elements_tetra = int(ss[id].split(' ')[-1])
        id += 1
        for i in range(id, id + n_elements_tetra):
            s = ss[i]
            tetra.append([int(x) for x in s.split(' ')[1:5]])
        id += n_elements_tetra
        print('debug 5 ', ss[id -1: id + 2])
        # if s.startswith('$Elements'):
        #     n_tetra = int(ss[ss.index(s) + 1])
        #     for i in range(n_tetra):
        #         s = ss[ss.index(s) + 2 + i]
        #         if s.split()[1] == '4':
        #             tetra.append([int(x) for x in s.split()[-4:]]) # 4 is the code for tetrahedron
    return nodes_tri, nodes_tetra, tetra


def vis_mesh(mesh, node_coords, node_coverage, pt_anchor, graph_edges, V_fixed_id_list, p_fixed_list, graph_nodes_control = None):

    from utils.vis import node_o3d_spheres, merge_meshes
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    node_mesh = node_o3d_spheres(node_coords, node_coverage * 0.1, color=[1, 0, 0])
    edges_pairs = []
    pcd_anchor = o3d.geometry.PointCloud()
    pcd_anchor.points = o3d.utility.Vector3dVector(node_coords[pt_anchor.reshape(-1).tolist()].tolist())
    pcd_anchor.paint_uniform_color([0, 1, 0])
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector([vertices[int(id)] + 1e-4 for id in V_fixed_id_list])
    pcd_s.paint_uniform_color([0, 0, 1])
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(p_fixed_list.detach().cpu().numpy().tolist())
    pcd_t.paint_uniform_color([1, 0, 0])

    pcd_gc = o3d.geometry.PointCloud()
    # pcd_gc.points = o3d.utility.Vector3dVector(graph_nodes_control.detach().cpu().numpy().tolist())
    pcd_gc.paint_uniform_color([0, 1, 1])
    print('pcd_gc', graph_nodes_control.shape)

    print('building line set')
    line_set = o3d.geometry.LineSet()
    for node_id, edges in enumerate(graph_edges):
        for neighbor_id in edges:
            if neighbor_id == -1:
                break
            edges_pairs.append([node_id, neighbor_id])
    from utils.line_mesh import LineMesh
    print('done line mesh')
    line_set.points = o3d.utility.Vector3dVector(node_coords.tolist())
    line_set.lines = o3d.utility.Vector2iVector(edges_pairs)
    y = -0.13
    r = 0.2
    center_shift = [0,0,-0.5]
    line_square = o3d.geometry.LineSet()
    line_square.points = o3d.utility.Vector3dVector(np.array([[-r, y, -r], [r, y, -r], [r, y, r], [-r, y, r], [-r, y, -r]]) + center_shift)
    line_square.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    # line_mesh = LineMesh(node_coords, edges_pairs, radius=0.0005)
    # edge_mesh = line_mesh.cylinder_segments
    # edge_mesh = merge_meshes(edge_mesh)
    # edge_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh, node_mesh, line_set], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([line_set, pcd_anchor, pcd_s, pcd_t, line_square, pcd_gc], mesh_show_back_face=True)

def inside(pts, lim):

    inside_id = (pts[:, 0] > lim[0][0]) * (pts[:, 0] < lim[1][0]) * (pts[:, 1] > lim[0][1]) * (pts[:, 1] < lim[1][1]) * (pts[:, 2] > lim[0][2]) * (pts[:, 2] < lim[1][2])
    print('inside', inside_id.sum())
    return inside_id

def paint(v_start, E_list, dis_coverage, v_uncovered, n_uncovered):

    from queue import PriorityQueue
    q = PriorityQueue()
    visit = set()
    dis = dict()
    q.put((0, v_start))
    cnt = 0
    while q.qsize() > 0:
        cnt += 1
        dis_u, u = q.get()
        visit.add(u)
        if u in v_uncovered:
            v_uncovered.remove(u)
            n_uncovered -= 1
        for e in E_list[u]:
            v = e[0]
            if v not in visit:
                new_dis = dis_u + e[1]
                if v not in dis or new_dis < dis[v]:
                    dis[v] = new_dis
                    if new_dis <= dis_coverage:
                        q.put((new_dis, v))
    print('cnt', cnt)
    return n_uncovered

def sample_nodes(v, faces, v_valid, coverage, use_only_valid = True, randomShuffle = True, vis_info = {}):

    # nodes_tri, nodes_tetra, tetra = msh_results
    # diff = torch.from_numpy(v) - torch.Tensor(nodes_tri)
    # diff = torch.norm(diff, dim=1)
    # print('diff', diff.min(), diff.max(), diff.mean())
    # exit(0)
    randomShuffle = True
    # E.shape (N, 2)
    E_list = [[] for i in range(v.shape[0])]
    # for i in range(E.shape[0]):
    #     e_len = np.linalg.norm(v[E[i, 0]] - v[E[i, 1]])
    #     E_list[E[i, 0]].append((E[i, 1], e_len))
    #     E_list[E[i, 1]].append((E[i, 0], e_len))
    for i in range(faces.shape[0]):
        f = faces[i]
        e_len = np.linalg.norm(v[f[0]] - v[f[1]])
        E_list[f[0]].append((f[1], e_len))
        E_list[f[1]].append((f[0], e_len))
        e_len = np.linalg.norm(v[f[1]] - v[f[2]])
        E_list[f[1]].append((f[2], e_len))
        E_list[f[2]].append((f[1], e_len))
        e_len = np.linalg.norm(v[f[2]] - v[f[0]])
        E_list[f[2]].append((f[0], e_len))
        E_list[f[0]].append((f[2], e_len))
    node_coords, node_indices = [], []
    v_uncovered = OrderedSet([i for i in range(v.shape[0])])
    n_uncovered = v.shape[0]
    if use_only_valid:
        for u in range(v.shape[0]):
            if not v_valid[u]:
                v_uncovered.remove(u)
                n_uncovered -= 1
    while n_uncovered > 0:
        if randomShuffle:
            v_id = np.random.randint(0, len(v_uncovered))
        else:
            v_id = 0
        v_id = v_uncovered[v_id]
        n_uncovered = paint(v_id, E_list, coverage, v_uncovered, n_uncovered)
        print(n_uncovered)
        node_coords.append(v[v_id])
        node_indices.append(v_id)

        if len(node_coords) == 100 and 0:

            mesh = vis_info['mesh']
            mesh.compute_vertex_normals()
            
            pcd_node = o3d.geometry.PointCloud()
            pcd_node.points = o3d.utility.Vector3dVector(node_coords)
            pcd_node.paint_uniform_color([1, 0, 0])
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector([v[u] for u in range(v.shape[0]) if u not in v_uncovered and u not in node_indices])
            pcd.paint_uniform_color([0, 1, 0])
            
            pcd_invalid = o3d.geometry.PointCloud()
            pcd_invalid.points = o3d.utility.Vector3dVector([v[u] for u in range(v.shape[0]) if not v_valid[u]])
            pcd_invalid.paint_uniform_color([0, 0, 1])
            
            o3d.visualization.draw_geometries([mesh, pcd, pcd_node, pcd_invalid])

    return np.array(node_coords), np.array(node_indices)

def get_deformation_graph_from_depthmap(depth_image, intrin, debug_mode=False, extra_info = {}):
    '''
    :param depth_image:
    :param intrin:
    :return:
    '''

    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion
    depth_scale = 1000.0
    max_triangle_distance = 0.04

    # Erosion of vertices in the boundaries
    EROSION_NUM_ITERATIONS = 0
    EROSION_MIN_NEIGHBORS = 0 # no erosion

    # Node sampling and edges computation
    node_coverage = 0.015  # in meters # 之前是0.01 for desk_open
    USE_ONLY_VALID_VERTICES = True
    num_neighbors = 8
    ENFORCE_TOTAL_NUM_NEIGHBORS = False
    SAMPLE_RANDOM_SHUFFLE = False

    # Pixel anchors
    NEIGHBORHOOD_DEPTH = 2

    MIN_CLUSTER_SIZE = 3
    MIN_NUM_NEIGHBORS = 1  # 2

    # Node clean-up
    # REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True

    suffix = ""

    # data_name = 'desk_open'
    # score_method = "block"
    # # score_method = "single"
    # choose_target_id = 13
    data_basename = extra_info['data_basename']
    score_method = extra_info['score_method']
    choose_target_id = extra_info['choose_target_id']
    src_time = extra_info['src_time']
    tgt_time = extra_info['tgt_time']
    img_depth_type = extra_info['img_depth_type']
    aspanformer_flag = extra_info['aspanformer_flag']
    target_depth_type = extra_info['target_depth_type']
    sigma = extra_info['sigma']
    score_thres = extra_info['score_thres']
    remove_iter = extra_info['remove_iter']
    mesh_type = extra_info['mesh_type']
    if mesh_type == "no_3d":
        remove_iter = 0
    finetune_id = extra_info['finetune_id']
    # task_name = f"{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_{score_method}_depth_gt_{depth_gt_flag}_aspanformer_{aspanformer_flag}"
    task_name = f"{data_basename}_{src_time}_{tgt_time}_{choose_target_id}_{score_method}_img_depth_type_{img_depth_type}_aspanformer_{aspanformer_flag}_target_depth_{target_depth_type}"
    if finetune_id != "99999":
        task_name += f"_finetune_id_{finetune_id}"
    print('task_name', task_name)
    data_name = f"{data_basename}_{src_time}"
    kpts_name = f"keypoints_{task_name}_{score_thres}_{remove_iter}.pth"
    
    pt_old_list, pt_new_list, id_list = torch.load(f"./kpts/{kpts_name}")
    camera_dir = f"/home/zt15/projects/nerfstudio/camera_info/{data_name}_train.pt"
    
    transform_matrix, c2w, scale_factor = torch.load(camera_dir)

    cam_intrinsics = {
        'fx': 1111.1,
        'fy': 1111.1,
        'cx': 400.,
        'cy': 400.,
        'width': 800,
        'height': 800,
    }

    canonical_cam_ = c2w[choose_target_id].cuda()
    canonical_cam = torch.eye(4).cuda()
    canonical_cam[:3] = canonical_cam_[:]
    canonical_cam[:,1] *= -1
    canonical_cam[:,2] *= -1

    #########################################################################
    """convert depth to mesh"""
    #########################################################################
    # width = depth_image.shape[1]
    # height = depth_image.shape[0]
    # mask_image=depth_image>0
    # # fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    # vertices, faces, vertex_pixels, point_image = depth_to_mesh(depth_image, mask_image, intrin, max_triangle_distance=max_triangle_distance, depth_scale=depth_scale)
    # num_vertices = vertices.shape[0]
    # num_faces = faces.shape[0]
    # assert num_vertices > 0 and num_faces > 0
    mesh_name = f"mesh_{sigma}_640_refined3"
    f_name_tri = f"./meshes/{data_name}_bb/{mesh_name}.obj"
    if not os.path.exists(f_name_tri):
        mesh_name = f"mesh_{sigma}_639_refined3"
        f_name_tri = f"./meshes/{data_name}_bb/{mesh_name}.obj"
    if not os.path.exists(f_name_tri):
        mesh_name = f"mesh_{sigma}_639_refined2"
        f_name_tri = f"./meshes/{data_name}_bb/{mesh_name}.obj"
    if not os.path.exists(f_name_tri):
        mesh_name = f"mesh_{sigma}_640_refined2"
        f_name_tri = f"./meshes/{data_name}_bb/{mesh_name}.obj"
    f_name_tetra_surface = f"./tetra_output/{data_name}.msh__sf.obj"
    f_name_tetra = f"./tetra_output/{data_name}.msh"
    mesh_original = o3d.io.read_triangle_mesh(f_name_tri)
    print('mesh', f_name_tri)
    # msh_results = read_msh_old(f_name_tri, f_name_tetra_surface, f_name_tetra)
    if mesh_type == 'tetra':
        nodes_tri, nodes_tetra, extra_triangle = read_msh(f_name_tri, f_name_tetra_surface, f_name_tetra)
    else:
        nodes_tri = None
        extra_triangle = torch.zeros(0, 3).long().cuda()
        nodes_tetra = torch.zeros(0, 3).cuda()
    # mesh_vis = o3d.io.read_triangle_mesh("./new_mesh.obj")
    # mesh_vis = mesh_original
    # mesh_door = o3d.io.read_triangle_mesh("./left_door.obj")
    # mesh_door_v = torch.Tensor(mesh_door.vertices)
    # door_lim = [mesh_door_v.min(0)[0], mesh_door_v.max(0)[0]]
    # door_lim[1][1] -= 0.02
    vertices = np.asarray(mesh_original.vertices)
    faces = np.asarray(mesh_original.triangles)

    # model_data = torch.load('./model_data.pth')
    # model_data = {
    #     **model_data,
    #     "mesh_original": mesh_original,
    # }
    # return model_data
    # pt_anchors_all = model_data["pixel_anchors_all"]
    # pt_weights_all = model_data["pixel_weights_all"]
    # V_fixed_id_list = model_data["V_fixed_id_list"]
    # p_fixed_list = model_data["p_fixed_list"]
    # graph_edges = model_data["graph_edges"].numpy()

    # pt_old_list, pixel_new_list, pt_new_list, id_list = torch.load("./keypoints_for_dg.pth")
    # min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    # V_fixed_id_list = id_list.cuda().long().squeeze(1)
    # V_fixed_id_list = V_fixed_id_list[:]
    # p_fixed_list = pt_new_list[:]
    # pt_anchors = pt_anchors_all[V_fixed_id_list]
    # pt_weights = pt_weights_all[V_fixed_id_list]
    # graph_nodes = model_data["graph_nodes"]
    # graph_nodes_control_bool = inside(graph_nodes, door_lim)
    
    # valid_p = ~torch.isnan(pt_weights.mean(-1))
    # print("pt_old_list", pt_old_list.max(0)[0], pt_old_list.min(0)[0])
    # for i in range(valid_p.shape[0]):
    #     if pt_old_list[i, 1] < -0.12:
    #         valid_p[i] = False
    # # valid_p[[661, 676, 40, 59]] = 1
    # valid_p[[676]] = 1
    # # valid_p[[86, 95, 105]] = 1
    
    # # valid_p2 = torch.zeros_like(valid_p)
    # # select_p = [86, 100, 200]
    # # valid_p2[select_p] = 1
    # # valid_p = valid_p & valid_p2
    # print(valid_p.sum(), pt_anchors.shape)
    # V_fixed_id_list = V_fixed_id_list[valid_p]
    # p_fixed_list = p_fixed_list[valid_p]
    # pt_anchors = pt_anchors[valid_p]
    # pt_weights = pt_weights[valid_p]
    # print("graph_nodes_control_bool", graph_nodes_control_bool.sum(), graph_nodes.shape)
    # vis_mesh(mesh_vis, graph_nodes.numpy(), node_coverage, pt_anchors, graph_edges, V_fixed_id_list, p_fixed_list, graph_nodes[graph_nodes_control_bool])
    # model_data["pixel_anchors"] = pt_anchors
    # model_data["pixel_weights"] = pt_weights
    # model_data["V_fixed_id_list"] = V_fixed_id_list
    # model_data["p_fixed_list"] = p_fixed_list
    # model_data["graph_nodes_control_bool"] = graph_nodes_control_bool
    # # exit(0)
    # return model_data
    
    # pt_anchor = model_data["pixel_anchors"]
    # pt_weight = model_data["pixel_weights"]
    # pt_anchor_all = model_data["pixel_anchors_all"]
    # pt_weight_all = model_data["pixel_weights_all"]
    
    # pt_old_list, pixel_new_list, pt_new_list, id_list = torch.load("./keypoints_for_dg.pth")
    # min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    # V_fixed_id_list = id_list.cuda().long().squeeze(1)
    
    # print('anchor and weight', pt_anchor, pt_weight)
    # print(pt_anchor_all[V_fixed_id_list[0]], pt_weight_all[V_fixed_id_list[0]])
    # input()
    # return model_data

    #########################################################################
    """Erode mesh, to not sample unstable nodes on the mesh boundary."""
    #########################################################################
    non_eroded_vertices = MVRegC.erode_mesh(vertices, faces, EROSION_NUM_ITERATIONS, EROSION_MIN_NEIGHBORS)




    #########################################################################
    """Sample graph nodes"""
    #########################################################################
    valid_vertices = non_eroded_vertices
    node_coords, node_indices = sample_nodes(vertices, faces, valid_vertices, node_coverage, use_only_valid = USE_ONLY_VALID_VERTICES, randomShuffle = SAMPLE_RANDOM_SHUFFLE, vis_info = {'mesh': mesh_original})
    # node_coords, node_indices = MVRegC.sample_nodes ( vertices, valid_vertices, node_coverage, USE_ONLY_VALID_VERTICES, SAMPLE_RANDOM_SHUFFLE)
    num_nodes = node_coords.shape[0]



    #########################################################################
    """visualize surface and non-eroded points"""
    #########################################################################
    if debug_mode:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[non_eroded_vertices.reshape(-1), :]))
        pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        print('node_coords', node_coords.shape)
        o3d.visualization.draw_geometries([mesh,  pcd_nodes], mesh_show_back_face=True)


    #########################################################################
    """Compute graph edges"""
    #########################################################################
    
    print('A', vertices.shape, valid_vertices.shape, faces.shape, node_indices.shape)
    
    vertices_tetra = nodes_tetra.cpu().numpy()
    # rotation = np.eye(4)
    # rotation[1:3, 1:3] = [[0, -1], [1, 0]]
    # vertices_tetra = (np.linalg.inv(rotation)[:3,:3] * vertices_tetra[:, None]).sum(-1)
    # tetra = np.array(msh_results[2])
    # print('debug graph edge', vertices.shape, valid_vertices.shape, faces.shape, node_indices.shape)
    # (540348, 3) (540348, 1) (1083941, 3) (2363,)
    # extra_triangle = []
    # for i in range(tetra.shape[0]):
    #     for j in range(4):
    #         for k in range(j+1, 4):
    #             for l in range(k+1, 4):
    #                 extra_triangle.append([tetra[i, j] - 1, tetra[i, k] - 1, tetra[i, l] - 1])
    # extra_triangle = np.array(extra_triangle)
    extra_triangle = extra_triangle.cpu().numpy()
    extra_valid = np.ones((vertices_tetra.shape[0], 1), dtype=bool)
    vertices = np.concatenate([vertices, vertices_tetra], axis=0)
    valid_vertices = np.concatenate([valid_vertices, extra_valid], axis=0)
    faces = np.concatenate([faces, extra_triangle], axis=0).astype(np.int32)
    # torch.save([vertices, faces], '../rsync/vertices_faces.pth')
    
    
    print(faces.min(), faces.max(), vertices.shape)
    graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances = \
        compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )
    
    vertices = vertices[:vertices.shape[0] - vertices_tetra.shape[0]]
    valid_vertices = valid_vertices[:valid_vertices.shape[0] - vertices_tetra.shape[0]]
    faces = faces[:faces.shape[0] - extra_triangle.shape[0]]
    node_to_vertex_distances = node_to_vertex_distances[:,:node_to_vertex_distances.shape[1] - vertices_tetra.shape[0]]
    
    print("AA", graph_edges.shape, graph_edges_weights.shape, graph_edges_distances.shape, node_to_vertex_distances.shape)
    print(vertices.shape, valid_vertices.shape, faces.shape, node_indices.shape)

    # graph_edges = MVRegC.compute_edges_euclidean(node_coords,num_neighbors, 0.05)
    # print(node_to_vertex_distances.shape, (node_to_vertex_distances[::1000] > 0).sum(-1)) # 比6多得多

    #########################################################################
    "Remove nodes"
    #########################################################################
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
    node_id_black_list = []
    if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
        MVRegC.node_and_edge_clean_up(graph_edges, valid_nodes_mask)
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    else:
        print("You're allowing nodes with not enough neighbors!")
    print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(node_id_black_list),
          "({})".format(node_id_black_list))

    #########################################################################
    """Compute pixel anchors"""
    #########################################################################
    # pixel_anchors = np.zeros((0), dtype=np.int32)
    # pixel_weights = np.zeros((0), dtype=np.float32) # 明天看这里是怎么计算到pixel_weights的，然后hack
    # MVRegC.compute_pixel_anchors_geodesic( node_to_vertex_distances, valid_nodes_mask, vertices, vertex_pixels, pixel_anchors, pixel_weights, width, height, node_coverage)
    # print("Valid pixels:", np.sum(np.all(pixel_anchors != -1, axis=2)))



    #########################################################################
    """filter invalid nodes"""
    #########################################################################
    node_coords = node_coords[valid_nodes_mask.squeeze()]
    node_indices = node_indices[valid_nodes_mask.squeeze()]
    graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    graph_edges_weights = graph_edges_weights[valid_nodes_mask.squeeze()]
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()]
    node_to_vertex_distances = node_to_vertex_distances[valid_nodes_mask.squeeze()]
    # print(node_to_vertex_distances[::1000].min(), node_to_vertex_distances[::1000].max(), 'debugg')

    #########################################################################
    """Check that we have enough nodes"""
    #########################################################################
    num_nodes = node_coords.shape[0]
    print(valid_nodes_mask.shape)
    print(valid_nodes_mask.sum())

    if (num_nodes == 0):
        print("No nodes! Exiting ...")
        exit()


    #########################################################################
    """Update node ids"""
    #########################################################################
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id] = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id] = np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edges[node_id, count_valid_neighbors] = mapped_neighbor_id
                    graph_edges_weights[node_id, count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                print("Hmmmmm", graph_edges_weights[node_id])
                raise Exception("Not good")

        # 3. Update pixel anchors using the id mapping (note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_geodesic_c' was given 'valid_nodes_mask')
        # MVRegC.update_pixel_anchors(node_id_mapping, pixel_anchors)

    pt_old_list, pt_new_list, id_list = torch.load(f"./kpts/{kpts_name}")
    min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    V_fixed_id_list = id_list.cuda().long().squeeze(1)
    p_fixed_list = pt_new_list.cuda()
    # V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list = remove_repeat(
    #     min_dis_all, V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list
    # )


    # V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list = (
    #     V_fixed_id_list[86:87],
    #     p_fixed_list[86:87],
    #     pt_old_list[86:87],
    #     pixel_new_list[86:87],
    # )


    # torch.save([V_fixed_id_list, num_neighbors, node_to_vertex_distances, node_coverage], "temp.pth")
    # pt_anchor, pt_weight = get_pt_anchor_weight(V_fixed_id_list, num_neighbors, node_to_vertex_distances, node_coverage)
    # # print('anchor and weight', pt_anchor, pt_weight)
    pt_anchor_all, pt_weight_all = get_pt_anchor_weight_batch(num_neighbors, node_to_vertex_distances, node_coverage)
    min_dis_mask = node_to_vertex_distances > node_coverage
    # print(pt_anchor, len(pt_anchor))
    # print(V_fixed_id_list, V_fixed_id_list.shape)
    # pt_anchor, pt_weight, pt_anchor_all, pt_weight_all, V_fixed_id_list = torch.load(f"temp_{data_name}_.pth")

    # print(pt_anchor_all.shape, pt_weight_all.shape, pt_anchor.shape, pt_weight.shape)
    # input('debug')
    # torch.save([pt_anchor, pt_weight, pt_anchor_all, pt_weight_all, V_fixed_id_list], f"temp_{data_name}_.pth")
    # pt_anchor = np.stack(pt_anchor)
    # pt_weight = np.stack(pt_weight)
    pt_anchor = pt_anchor_all[V_fixed_id_list]
    pt_weight = pt_weight_all[V_fixed_id_list]
    # print(pt_anchor_all.shape, pt_weight_all.shape, pt_anchor_all[V_fixed_id_list[0]], pt_weight_all[V_fixed_id_list[0]])
    # torch.save([pt_anchor, pt_weight, pt_anchor_all, pt_weight_all, V_fixed_id_list], "temp.pth")

    # print(pt_anchor_all, pt_weight_all)

    #########################################################################
    """Compute clusters."""
    #########################################################################
    graph_clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32)
    clusters_size_list = MVRegC.compute_clusters(graph_edges, graph_clusters)
    print("clusters_size_list", clusters_size_list)


    #########################################################################
    """visualize valid pixels"""
    #########################################################################
    # if debug_mode and 0:
    #     from utils.vis import save_grayscale_image
    #     pixel_anchors_image = np.sum(pixel_anchors, axis=2)
    #     pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
    #     raw_pixel_mask = np.copy(pixel_anchors_image).astype(np.uint8) * 0
    #     pixel_anchors_mask[pixel_anchors_image == -4] = 0
    #     raw_pixel_mask[depth_image > 0] = 1
    #     pixel_anchors_mask[pixel_anchors_image > -4] = 1
    #     save_grayscale_image("./output/pixel_anchors_mask.jpeg", pixel_anchors_mask)
    #     save_grayscale_image("./output/depth_mask.jpeg", raw_pixel_mask)


    #########################################################################
    """visualize graph"""
    #########################################################################

    # print(point_image.shape)
    # input('aha')
    model_data = {
        "graph_nodes": torch.from_numpy( node_coords),
        "node_indices": torch.from_numpy( node_indices),
        "graph_edges": torch.from_numpy( graph_edges).long(),
        "graph_edges_weights": torch.from_numpy( graph_edges_weights),
        "graph_clusters": graph_clusters,
        "pixel_anchors": torch.from_numpy(np.array(pt_anchor)),
        "pixel_weights": torch.from_numpy(np.array(pt_weight)),
        "pixel_anchors_all": pt_anchor_all,
        "pixel_weights_all": pt_weight_all,
        "V_fixed_id_list": V_fixed_id_list,
        "p_fixed_list": p_fixed_list,
        "min_dis_mask": torch.from_numpy(min_dis_mask),
        "data_name": f"{task_name}",
        "data_basename_src": f"{data_basename}_{src_time}",
        "canonical_cam": canonical_cam,
        "cam_intrinsics": cam_intrinsics,
        # "point_image": torch.from_numpy(point_image).permute(1,2,0) # [H, W, 3]
    }
    # torch.save(model_data, './model_data.pth')
    model_data = {
        **model_data,
        "mesh_original": mesh_original,
    }

    if debug_mode:
        from utils.vis import node_o3d_spheres, merge_meshes

        node_mesh = node_o3d_spheres(node_coords, node_coverage * 0.1, color=[1, 0, 0])
        edges_pairs = []
        pcd_anchor = o3d.geometry.PointCloud()
        pcd_anchor.points = o3d.utility.Vector3dVector([node_coords[int(id)] for id in pt_anchor[0]])
        pcd_anchor.paint_uniform_color([0, 1, 0])
        pcd_st = o3d.geometry.PointCloud()
        pcd_st.points = o3d.utility.Vector3dVector([vertices[int(id)] for id in V_fixed_id_list] + p_fixed_list.detach().cpu().numpy().tolist())
        pcd_st.paint_uniform_color([0, 0, 1])
        print('building line set')
        line_set = o3d.geometry.LineSet()
        for node_id, edges in enumerate(graph_edges):
            for neighbor_id in edges:
                if neighbor_id == -1:
                    break
                edges_pairs.append([node_id, neighbor_id])
        from utils.line_mesh import LineMesh
        print('done line mesh')
        line_set.points = o3d.utility.Vector3dVector(node_coords.tolist())
        line_set.lines = o3d.utility.Vector2iVector(edges_pairs)

        line_corr = o3d.geometry.LineSet()
        line_corr.points = o3d.utility.Vector3dVector(pt_old_list.cpu().numpy().tolist() + pt_new_list.cpu().numpy().tolist())
        line_corr.lines = o3d.utility.Vector2iVector([[i, i + len(pt_old_list)] for i in range(len(pt_old_list))])
        line_corr.paint_uniform_color([1, 0, 0])
        # line_mesh = LineMesh(node_coords, edges_pairs, radius=0.0005)
        # edge_mesh = line_mesh.cylinder_segments
        # edge_mesh = merge_meshes(edge_mesh)
        # edge_mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh, node_mesh, line_set], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([line_set, pcd_anchor, pcd_st, line_corr], mesh_show_back_face=True)

    return model_data


def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]



def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def multual_nn_correspondence(src_pcd_deformed, tgt_pcd, search_radius=0.3, knn=1):

    src_idx = np.arange(src_pcd_deformed.shape[0])

    s2t_dists, ref_tgt_idx = knn_point_np (knn, tgt_pcd, src_pcd_deformed)
    s2t_dists, ref_tgt_idx = s2t_dists[:,0], ref_tgt_idx [:, 0]
    valid_distance = s2t_dists < search_radius

    _, ref_src_idx = knn_point_np (knn, src_pcd_deformed, tgt_pcd)
    _, ref_src_idx = _, ref_src_idx [:, 0]

    cycle_src_idx = ref_src_idx [ ref_tgt_idx ]

    is_mutual_nn = cycle_src_idx == src_idx

    mutual_nn = np.logical_and( is_mutual_nn, valid_distance)
    correspondences = np.stack([src_idx [ mutual_nn ], ref_tgt_idx[mutual_nn] ] , axis=0)

    return correspondences