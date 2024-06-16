import torch
import numpy as np
import MVRegC
import open3d as o3d




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
    # R.shape: [N,6,3,3] x: [N,1,3] - [N,6,3] -> 
    # print('dd', R.shape, x.shape, g.shape)
    # print('debug', (R * (x[:,None] - g)[:,:,None]).sum(-1).shape, g.shape, t.shape, w.shape)
    # print('debb', x.mean(), g.mean(), t.mean(), w.mean())
    y = ( (R * (x[:,None] - g)[:,:,None]).sum(-1) + g + t ) * w[...,None]
    # y = (R @ (x - g) + g + t ) * w （注意这里有两种g的表示，可以这样，也可以使用原本vertex的所在地方）
    # y = ( R * (x[:,None] - g) + g + t ) * w[...,None]
    y = y.sum(dim=1) # [N,3]
    return y




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


def argmin_dis_batch(p, q, K=1):
    print("argmin_dis_batch (pytorch3d)", p.shape, q.shape)
    from pytorch3d import ops

    result = ops.knn_points(p[None], q[None], K=K)

    return result[0][0, :, :], result[1][0, :, :]

def get_pt_anchor_weight(vertices_id_list, n_neighbor, node_vertex_dis, node_coverage):

    pt_anchor = []
    pt_weight = []
    for v in vertices_id_list:
        dis_and_id = []
        # print('in 1', v, node_vertex_dis[:,v].sum())
        for node in range(node_vertex_dis.shape[0]):
            dis = node_vertex_dis[node, v]
            if dis < 0:
                continue
            dis_and_id.append([dis, node])
        dis_and_id = sorted(dis_and_id, key=lambda x: x[0])
        pt_anchor.append([x[1] for x in dis_and_id[:n_neighbor]])
        for j in range(n_neighbor - len(pt_anchor[-1])):
            pt_anchor[-1].append(-1)
        weight = np.array([x[0] for x in dis_and_id[:n_neighbor]])
        # print('w in 1', weight)
        weight = np.exp(- (weight * weight) / (2 * node_coverage * node_coverage))
        weight = weight / np.sum(weight)
        pt_weight.append(weight)
    
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
        with torch.no_grad():
            anchor, weight = get_pt_anchor_weight_minibatch(n_neighbor, vertex_node_dis[i:i+bs], node_coverage)
            pt_anchor.append(anchor.cpu())
            pt_weight.append(weight.cpu())
    pt_anchor = torch.cat(pt_anchor, axis=0)
    pt_weight = torch.cat(pt_weight, axis=0)
    return pt_anchor, pt_weight

    pt_old_list, pixel_new_list, pt_new_list, id_list = torch.load("./keypoints_for_dg.pth")
    min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    V_fixed_id_list = id_list.cuda().long().squeeze(1)
    print(V_fixed_id_list.shape, id_list[0])
    p_fixed_list = pt_new_list.cuda()

    # V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list = remove_repeat(
    #     min_dis_all, V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list
    # )
    V_fixed_id_list, p_fixed_list, pt_old_list, pixel_new_list = (
        V_fixed_id_list[86:87],
        p_fixed_list[86:87],
        pt_old_list[86:87],
        pixel_new_list[86:87],
    )
    # torch.save([V_fixed_id_list, num_neighbors, node_to_vertex_distances, node_coverage], "temp.pth")
    pt_anchor, pt_weight = get_pt_anchor_weight(V_fixed_id_list, num_neighbors, node_to_vertex_distances, node_coverage)
    # print('anchor and weight', pt_anchor, pt_weight)
    pt_anchor_all, pt_weight_all = get_pt_anchor_weight_batch(num_neighbors, node_to_vertex_distances, node_coverage)
    
    # print(pt_anchor_all.shape, pt_weight_all.shape, pt_anchor_all[V_fixed_id_list[0]], pt_weight_all[V_fixed_id_list[0]])
    # torch.save([pt_anchor, pt_weight, pt_anchor_all, pt_weight_all, V_fixed_id_list], "temp.pth")

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

def get_deformation_graph_from_depthmap (depth_image, intrin, debug_mode=True):
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
    node_coverage = 0.01  # in meters
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

    #########################################################################
    """convert depth to mesh"""
    #########################################################################
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    mask_image=depth_image>0
    # fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    vertices, faces, vertex_pixels, point_image = depth_to_mesh(depth_image, mask_image, intrin, max_triangle_distance=max_triangle_distance, depth_scale=depth_scale)
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    assert num_vertices > 0 and num_faces > 0
    mesh_original = o3d.io.read_triangle_mesh("./meshes/mesh_128_200_lap.obj")
    mesh_vis = o3d.io.read_triangle_mesh("./new_mesh.obj")
    mesh_vis = mesh_original
    mesh_door = o3d.io.read_triangle_mesh("./left_door.obj")
    mesh_door_v = torch.Tensor(mesh_door.vertices)
    door_lim = [mesh_door_v.min(0)[0], mesh_door_v.max(0)[0]]
    door_lim[1][1] -= 0.02
    vertices = np.asarray(mesh_original.vertices)
    faces = np.asarray(mesh_original.triangles)

    model_data = torch.load('./model_data.pth')
    model_data = {
        **model_data,
        "mesh_original": mesh_original,
    }
    # return model_data
    pt_anchors_all = model_data["pixel_anchors_all"]
    pt_weights_all = model_data["pixel_weights_all"]
    V_fixed_id_list = model_data["V_fixed_id_list"]
    p_fixed_list = model_data["p_fixed_list"]
    graph_edges = model_data["graph_edges"].numpy()

    pt_old_list, pixel_new_list, pt_new_list, id_list = torch.load("./keypoints_for_dg.pth")
    min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    V_fixed_id_list = id_list.cuda().long().squeeze(1)
    V_fixed_id_list = V_fixed_id_list[:]
    p_fixed_list = pt_new_list[:]
    pt_anchors = pt_anchors_all[V_fixed_id_list]
    pt_weights = pt_weights_all[V_fixed_id_list]
    graph_nodes = model_data["graph_nodes"]
    graph_nodes_control_bool = inside(graph_nodes, door_lim)
    
    valid_p = ~torch.isnan(pt_weights.mean(-1))
    print("pt_old_list", pt_old_list.max(0)[0], pt_old_list.min(0)[0])
    for i in range(valid_p.shape[0]):
        if pt_old_list[i, 1] < -0.12:
            valid_p[i] = False
    # valid_p[[661, 676, 40, 59]] = 1
    valid_p[[676]] = 1
    # valid_p[[86, 95, 105]] = 1
    
    # valid_p2 = torch.zeros_like(valid_p)
    # select_p = [86, 100, 200]
    # valid_p2[select_p] = 1
    # valid_p = valid_p & valid_p2
    print(valid_p.sum(), pt_anchors.shape)
    V_fixed_id_list = V_fixed_id_list[valid_p]
    p_fixed_list = p_fixed_list[valid_p]
    pt_anchors = pt_anchors[valid_p]
    pt_weights = pt_weights[valid_p]
    print("graph_nodes_control_bool", graph_nodes_control_bool.sum(), graph_nodes.shape)
    vis_mesh(mesh_vis, graph_nodes.numpy(), node_coverage, pt_anchors, graph_edges, V_fixed_id_list, p_fixed_list, graph_nodes[graph_nodes_control_bool])
    model_data["pixel_anchors"] = pt_anchors
    model_data["pixel_weights"] = pt_weights
    model_data["V_fixed_id_list"] = V_fixed_id_list
    model_data["p_fixed_list"] = p_fixed_list
    model_data["graph_nodes_control_bool"] = graph_nodes_control_bool
    # exit(0)
    return model_data
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
    node_coords, node_indices = MVRegC.sample_nodes ( vertices, valid_vertices, node_coverage, USE_ONLY_VALID_VERTICES, SAMPLE_RANDOM_SHUFFLE)
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
    graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances = \
        compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )
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
    print(node_to_vertex_distances[::1000].min(), node_to_vertex_distances[::1000].max(), 'debugg')

    #########################################################################
    """Check that we have enough nodes"""
    #########################################################################
    num_nodes = node_coords.shape[0]
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

    pt_old_list, pixel_new_list, pt_new_list, id_list = torch.load("./keypoints_for_dg.pth")
    min_dis_all, id_list = argmin_dis_batch(pt_old_list.cuda(), torch.Tensor(vertices).cuda())
    print(min_dis_all.mean())
    V_fixed_id_list = id_list.cuda().long().squeeze(1)
    print(V_fixed_id_list.shape, id_list[0])
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
    # print('anchor and weight', pt_anchor, pt_weight)
    pt_anchor_all, pt_weight_all = get_pt_anchor_weight_batch(num_neighbors, node_to_vertex_distances, node_coverage)
    pt_anchor = pt_anchor[V_fixed_id_list]
    pt_weight = pt_weight[V_fixed_id_list]
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
    if debug_mode and 0:
        from utils.vis import save_grayscale_image
        pixel_anchors_image = np.sum(pixel_anchors, axis=2)
        pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
        raw_pixel_mask = np.copy(pixel_anchors_image).astype(np.uint8) * 0
        pixel_anchors_mask[pixel_anchors_image == -4] = 0
        raw_pixel_mask[depth_image > 0] = 1
        pixel_anchors_mask[pixel_anchors_image > -4] = 1
        save_grayscale_image("./output/pixel_anchors_mask.jpeg", pixel_anchors_mask)
        save_grayscale_image("./output/depth_mask.jpeg", raw_pixel_mask)


    #########################################################################
    """visualize graph"""
    #########################################################################

    print(point_image.shape)
    # input('aha')
    model_data = {
        "graph_nodes": torch.from_numpy( node_coords),
        "graph_edges": torch.from_numpy( graph_edges).long(),
        "graph_edges_weights": torch.from_numpy( graph_edges_weights),
        "graph_clusters": graph_clusters,
        "pixel_anchors": torch.from_numpy(np.array(pt_anchor)),
        "pixel_weights": torch.from_numpy(np.array(pt_weight)),
        "pixel_anchors_all": pt_anchor_all,
        "pixel_weights_all": pt_weight_all,
        "V_fixed_id_list": V_fixed_id_list,
        "p_fixed_list": p_fixed_list,
        # "point_image": torch.from_numpy(point_image).permute(1,2,0) # [H, W, 3]
    }
    torch.save(model_data, './model_data.pth')
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
        # line_mesh = LineMesh(node_coords, edges_pairs, radius=0.0005)
        # edge_mesh = line_mesh.cylinder_segments
        # edge_mesh = merge_meshes(edge_mesh)
        # edge_mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh, node_mesh, line_set], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([mesh, line_set, pcd_anchor, pcd_st], mesh_show_back_face=True)

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