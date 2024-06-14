import json
import math
import os
import sys
from math import radians

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

DEBUG = False

VIEWS = 440
val_VIEWS = 1
# VIEWS = 10
# val_VIEWS = 10
res_scale = 3.6
RESOLUTION = int(800 * res_scale)
# DEPTH_SCALE = 1.4
FORMAT = "PNG"
COLOR_DEPTH = 8
RANDOM_VIEWS = True
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0.3, 0, 0)

FIX_SEED = True
if FIX_SEED:
    import random

    random.seed(2)
    np.random.seed(2)

# NERF deformation parameters
# scale [default: 2.2]
SCALE = 2.2
# rotation: in degree
ROTATION_X = 0
ROTATION_Y = 0
ROTATION_Z = 0
# translation
TRANSLATION_X = 0
TRANSLATION_Y = 0
TRANSLATION_Z = 0

# data_name = "zebra_30" # "zebra_12" "zebra_20"
# data_name = "phoenix_1" # "phoenix_2" "phoenix_15"
# data_name = "yetiseal_0" # "yetiseal_17"
# data_name = "claris_15" # "claris_17" "claris_19" "claris_26"
# data_name = "dragon_14" # "dragon_17" "dragon_30"
# data_name = "desk3_0" # "desk3_22" "desk3_42"
# data_name = "desk4_0" # "desk3_20" "desk3_40"
# data_name = "robot_steel_1" # "robot_steel_2" "robot_steel_4" "robot_steel_10"
data_name = "phoenix2_22"  # "phoenix2_8" "phoenix2_10" "phoenix2_22"

start_pose = 3
file_name = sys.argv[start_pose + 1]
time_id = int(sys.argv[start_pose + 2])
scene_config = json.load(open(rf"./scene_config.json", "r"))
data_name = f"{file_name}_{time_id}"


uri_flag = False
if len(file_name) > 20:
    uri_flag = True

scale = scene_config[file_name]["scale"]
translate = scene_config[file_name][str(time_id)]["translate"]
translate[2], translate[1] = translate[1], -translate[2]

INPUT_OBJ_FN = rf"./glbs/{file_name}.glb"
RESULTS_PATH = f"./{data_name}_high"

bb_original = np.array([[-1.0, -1, -1], [1.0, 1, 1]])
bb_original_mean = bb_original.mean(0)
bb_original = (bb_original - bb_original_mean) * 1.2 + bb_original_mean
bb_original = bb_original.tolist()

f = open("C:/Users/tangz/Desktop/blender/cmc-render-main/output.txt", "w")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting
    bpy.context.scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty


def generate_y_rot():
    threshold = -(np.pi / 6 * 40 / 30)
    while 1:
        angle = np.arccos(1 - 2 * np.random.rand()) - np.pi / 2
        if angle >= threshold:
            break
    return angle


# camera
def get_calibration_matrix_K_from_blender(camd):
    """From DeformingThings4D"""
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        s_u = s_v / pixel_aspect_ratio
    else:  # 'HORIZONTAL' and 'AUTO'
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = s_u / pixel_aspect_ratio
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels
    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


def opencv_to_blender(T):
    """T: ndarray 4x4
    usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, origin)


def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return np.matmul(T, transform)


def Print(x, f):
    print(*x, file=f, flush=True)


def set_cycles_renderer(
    scene: bpy.types.Scene,
    camera_object: bpy.types.Object,
    num_samples: int,
    use_denoising: bool = True,
    use_motion_blur: bool = False,
    use_transparent_bg: bool = False,
    prefer_cuda_use: bool = True,
    use_adaptive_sampling: bool = False,
) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "CYCLES"
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    scene.cycles.samples = num_samples

    # Enable GPU acceleration
    # Source - https://blender.stackexchange.com/a/196702
    if prefer_cuda_use:
        bpy.context.scene.cycles.device = "GPU"

        # Change the preference setting
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {}".format(d["name"]))
    print("----")


def main():
    fp = bpy.path.abspath(f"//{RESULTS_PATH}")
    os.makedirs(fp, exist_ok=True)

    # Data to store in JSON file
    out_data = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
        "fl_x": 1111.1 * res_scale,
        "fl_y": 1111.1 * res_scale,
        "cx": int(400 * res_scale),
        "cy": int(400 * res_scale),
        "w": RESOLUTION,
        "h": RESOLUTION,
    }

    val_out_data = {
        "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
    }

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    # bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_diffuse_color = True
    bpy.context.scene.render.image_settings.file_format = str(FORMAT)
    bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

    # Create input render layer node.
    render_layers = tree.nodes.new("CompositorNodeRLayers")

    # # Create depth output nodes
    # depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # depth_file_output.label = 'Depth Output'
    # if FORMAT == 'OPEN_EXR':
    #     links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    # else:
    #     depth_file_output.format.color_mode = 'BW'
    #     # Remap as other types can not represent the full range of depth.
    #     map_value_node = tree.nodes.new(type="CompositorNodeMapValue")
    #     # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    #     # map_value_node.offset = [-0.7]
    #     map_value_node.size = [DEPTH_SCALE]
    #     map_value_node.use_min = True
    #     map_value_node.min = [0]

    #     # g_depth_clip_start = 0.5
    #     # g_depth_clip_end = 4
    #     # map_value_node.offset[0] = -g_depth_clip_start
    #     # map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
    #     # map_value_node.use_min = True
    #     # map_value_node.use_max = True
    #     # map_value_node.min[0] = 0.0
    #     # map_value_node.max[0] = 1.0

    #     # links.new(render_layers.outputs['Depth'], map_value_node.inputs[0])
    #     links.new(render_layers.outputs[2], map_value_node.inputs[0])
    #     links.new(map_value_node.outputs[0], depth_file_output.inputs[0])

    # # Create normal output nodes
    # scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    # scale_normal.blend_type = 'MULTIPLY'
    # # scale_normal.use_alpha = True
    # scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    # links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    # bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    # bias_normal.blend_type = 'ADD'
    # # bias_normal.use_alpha = True
    # bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    # links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    # normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # normal_file_output.label = 'Normal Output'
    # links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

    # albedo
    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    links.new(render_layers.outputs["DiffCol"], albedo_file_output.inputs[0])

    # Create collection for objects not to render with background
    # Delete default cube
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Cube"].select_set(state=True)
    bpy.ops.object.delete(use_global=False)

    # Import textured mesh
    # bpy.ops.import_scene.obj(filepath=INPUT_OBJ_FN, axis_forward='Y', axis_up='Z')
    print(INPUT_OBJ_FN)
    bpy.ops.import_scene.gltf(filepath=INPUT_OBJ_FN)
    print(scale)
    location_sum = (
        np.abs(bpy.context.object.location[0])
        + np.abs(bpy.context.object.location[1])
        + np.abs(bpy.context.object.location[2])
    )
    # if location_sum < 1e-8:
    #     print("already centered")
    #     exit()
    bpy.context.object.location[0] = 0
    bpy.context.object.location[1] = 0
    bpy.context.object.location[2] = 0
    bpy.context.object.scale[0] *= scale
    bpy.context.object.scale[1] *= scale
    bpy.context.object.scale[2] *= scale
    bpy.ops.transform.translate(value=translate)
    bpy.data.scenes["Scene"].frame_current = time_id

    # bpy.ops.import_scene.gltf(filepath=INPUT_OBJ_FN)
    # # mesh_obj = bpy.context.selected_objects[0]
    # for x in bpy.context.selected_objects:
    #     Print([x.dimensions], f)
    # mesh_obj = bpy.data.objects['Sketchfab_model']

    # f.write(str(list(bpy.data.objects)))

    # bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")
    # bpy.context.view_layer.objects.active = mesh_obj

    # # scale
    # factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / SCALE

    # f.write("\n" + str(factor) + " " + str(SCALE) + " " + str(mesh_obj.dimensions))
    # f.flush()
    # mesh_obj.scale[0] /= factor
    # mesh_obj.scale[1] /= factor
    # mesh_obj.scale[2] /= factor
    # bpy.ops.object.transform_apply(scale=True)

    # rotation
    # bpy.context.active_object.rotation_euler[0] = math.radians(ROTATION_X)
    # bpy.context.active_object.rotation_euler[1] = math.radians(ROTATION_Y)
    # bpy.context.active_object.rotation_euler[2] = math.radians(ROTATION_Z)

    # translation
    # bpy.context.active_object.delta_location = (float(TRANSLATION_X), float(TRANSLATION_Y), float(TRANSLATION_Z))

    # light

    energy = 70.0

    light = bpy.data.lights["Light"]
    light.energy = energy
    bpy.data.objects["Light"].location = (0, 2.0, 1)

    bpy.ops.object.light_add(type="POINT")
    bpy.data.lights["Point"].energy = energy
    bpy.data.objects["Point"].location = (2.0, 0, 1)

    bpy.ops.object.light_add(type="POINT")
    bpy.data.lights["Point.001"].energy = energy
    bpy.data.objects["Point.001"].location = (0, -2.0, 1)

    bpy.ops.object.light_add(type="POINT")
    bpy.data.lights["Point.002"].energy = energy
    bpy.data.objects["Point.002"].location = (-2, 0.0, 1)

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 3
    # light.use_shadow = False
    # # Possibly disable specular shading:
    # light.specular_factor = 1.0

    # # Add another light source so stuff facing away from light is not completely dark
    # bpy.ops.object.light_add(type='POINT')
    print("lights", list(bpy.data.lights.keys()), list(bpy.data.objects.keys()))
    # light_2 = bpy.data.lights['Sun']
    # light_2.use_shadow = False
    # light_2.specular_factor = 1.0
    # light_2.energy = 0
    # # bpy.data.objects['Sun'].location = (0, 100, 0)
    # bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    # bpy.data.objects['Sun'].rotation_euler[0] += 180

    # rendering
    bpy.context.scene.render.use_placeholder = False
    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = RESOLUTION
    bpy.context.scene.render.resolution_y = RESOLUTION
    bpy.context.scene.render.resolution_percentage = 100

    cam = bpy.context.scene.objects["Camera"]
    cam.location = (0, 4, 0.5)
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    K = get_calibration_matrix_K_from_blender(cam.data)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    """dump intrinsics & extrinsics"""
    np.savetxt(fp + "/cam_intr.txt", np.array(K))

    # for output_node in [normal_file_output, albedo_file_output, depth_file_output]:
    for output_node in [albedo_file_output]:
        output_node.base_path = ""
        output_node.format.file_format = FORMAT

    out_data["frames"] = []
    val_out_data["frames"] = []

    if not RANDOM_VIEWS:
        b_empty.rotation_euler = CIRCLE_FIXED_START

    stepsize = 360.0 / VIEWS
    for i in range(VIEWS + val_VIEWS):
        if RANDOM_VIEWS:
            bpy.context.scene.render.filepath = fp + "/%06d" % i
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
                if uri_flag:
                    rot[0] = generate_y_rot()
                else:
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            bpy.context.scene.render.filepath = fp + "/%06d" % i

        # depth_file_output.file_slots[0].path = bpy.context.scene.render.filepath + "_depth_"
        # normal_file_output.file_slots[0].path = bpy.context.scene.render.filepath + "_normal_"
        # albedo_file_output.file_slots[0].path = bpy.context.scene.render.filepath + "_albedo_"

        # render
        bpy.ops.render.render(write_still=True)  # render still

        # os.rename(bpy.context.scene.render.filepath + "_albedo_" + "0001.png",
        #     bpy.context.scene.render.filepath + "_albedo.png")

        """compute extrinsics"""
        cam_blender = np.array(cam.matrix_world)
        cam_opencv = blender_to_opencv(cam_blender)
        # np.savetxt(bpy.context.scene.render.filepath + '_cam_ext.txt' , cam_opencv)

        # ## compute ray
        # u, v = np.meshgrid(range(RESOLUTION), range(RESOLUTION))
        # u = u.reshape(-1)
        # v = v.reshape(-1)
        # pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        # cam_rotation = cam_opencv[:3, :3]
        # pix_position = np.matmul(cam_rotation, pix_position.transpose()).transpose()
        # ray_direction = pix_position / np.linalg.norm(pix_position, axis=1, keepdims=True)
        # ray_origin = cam_opencv[:3, 3:].transpose()

        # """explicitly cast rays to get point cloud """
        # ray_begin_local = mesh_obj.matrix_world.inverted() @ Vector(ray_origin[0])
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        # bvhtree = BVHTree.FromObject(mesh_obj, depsgraph)
        # pcl = np.zeros_like(ray_direction)
        # for _j in range(ray_direction.shape[0]):
        #     position, norm, faceID, _  = bvhtree.ray_cast(ray_begin_local, Vector(ray_direction[_j]), 200)
        #     if position: # hit a triangle
        #         pcl[_j]= Matrix(cam_opencv).inverted() @ mesh_obj.matrix_world @ position

        # """dump depth"""
        # depth = pcl[:,2].reshape((RESOLUTION, RESOLUTION))
        # depth = (depth*1000).astype(np.uint16) #  resolution 1mm
        # np.save(bpy.context.scene.render.filepath + "_depth.npy", depth)
        frame_data = {
            "file_path": os.path.basename(bpy.context.scene.render.filepath) + ".png",
            "rotation": radians(stepsize),
            "transform_matrix": listify_matrix(cam.matrix_world),
        }
        if i >= VIEWS:
            val_out_data["frames"].append(frame_data)
        else:
            out_data["frames"].append(frame_data)

        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
        else:
            b_empty.rotation_euler[2] += radians(stepsize)

    out_data["bb_original"] = bb_original
    val_out_data["bb_original"] = bb_original

    with open(fp + "/" + "transforms.json", "w") as out_file:
        json.dump(out_data, out_file, indent=4)

    with open(fp + "/" + "transforms_val.json", "w") as out_file:
        json.dump(val_out_data, out_file, indent=4)

    with open(fp + "/" + "transforms_test.json", "w") as out_file:
        json.dump(val_out_data, out_file, indent=4)

    f.flush()


if __name__ == "__main__":
    main()
