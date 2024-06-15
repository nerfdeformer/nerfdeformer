## Dataset

The full data can be downloaded [here](https://drive.google.com/drive/folders/1n1J_eS2AbPMQ0Tpf53ZBkV5mZ7zbGnNP?usp=sharing).

The data folder is structured as below:

### scene config

`scene_config.json` defines all original-transformed scene pairs, which is a list of dict. one dict defines one pair. For example:

```python
    "obj_A": {
        "src_tgt_pair": [
            [
                0,
                3
            ],
            [
                0,
                22
            ],
            [
                0,
                34
            ]
        ]
    }
```

means the dynamic object `obj_A` forms three original-transformed scene pairs: time=0 or 3 as the original/transformed scene, time=0 or 22, time=0 or 34.

### frames

`frames/original_nerf_views/{s}_{t}/{id}.png`: frames of the object `s` at the original time `t`.
`frames/original_nerf_views/{s}_{t}/transforms.json`: camera parameters of original frames.

`frames/frames/transformed_views/{s}_{t2}/transformed_view/{id}.png`: transformed rgb view of the object `s` at the transformed time `t2`.
`frames/frames/transformed_views/{s}_{t2}/transformed_view/{id}_depth.npy`: transformed depth view.
`frames/frames/transformed_views/{s}_{t2}/transformed_view/transforms.json`: camera parameters of the transformed view.

`frames/frames/transformed_views/{s}_{t2}/transformed_view/{id}.png`: transformed rgb view of the transformed object `s` at time `t2`.
`frames/frames/transformed_views/{s}_{t2}/transformed_view/{id}_depth.npy`: transformed depth view.
`frames/frames/transformed_views/{s}_{t2}/transformed_view/transforms.json`: camera parameters of the transformed view.

`frames/frames/transformed_views/{s}_{t2}/evaluations/`: 30 other rgbd views and camera parameters rendered surrounding `s` at time `t2`, which are what used to evaluate in our paper.

### glbs

glb files for the dynamic objects.

### meshes

obj files for dynamic objects in the time defined in `scene_config.json`.

### rendering code

`blender.py` is a sample script to render views surrounding the object loadinging glb files and specifying the time. You can change it to render your own.