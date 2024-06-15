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

### rendering code

blender.py is a sample script to render views surrounding the object loading glb files and specify the timestep. you can change it to render your own.