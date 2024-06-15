## Dataset

The full data can be downloaded [here](https://drive.google.com/drive/folders/1n1J_eS2AbPMQ0Tpf53ZBkV5mZ7zbGnNP?usp=sharing).

The data folder is structured as below:

### scene config

    `scene_config.json` defines each scene, which is a list of dict. For example:
    '''python
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
    },
    '''
    means the dynamic object `obj_A` forms three scenes: time=0 as the original one and time=3 as the transformed one; time=0 as the original one and time=22 as the transformed one; time=0 as the original one and time=34 as the transformed one.

### rendering code

blender.py is a sample script to render our data loading glb files. you can change it to render your own.