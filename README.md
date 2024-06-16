# NeRFDeformer

## Data

please check [data](https://github.com/nerfdeformer/nerfdeformer/blob/main/data) for more datails.

## our evaluations

we put our full new view synthesis and geometry results in [our_evaluations](https://github.com/nerfdeformer/nerfdeformer/blob/main/our_evaluations).

## code

The code is separated to three parts: dense correspondence **matching**, correspondence **filtering**, **embedded deformation graph** (EDG) optimization, NeRF **new view synthesis in transformed scene** using EDG as deformation field. We have a data_sample for inference

They are located in [src/correspondence_matching](), [src/EDG]() and [src/NeRF]().

### correspondence matching

Our corrspondence matching code is based on [ASpanFormer](https://github.com/apple/ml-aspanformer). We also use its outdoor pretrained model for inference. We include the pretrained model in the repository. 

To install the conda virtual environment, run `conda install --file src/correspondence_matching/environment.yaml`.

Then run the following commands to do matching on the sample data.

```python
cd src/correspondence_matching/demo

python match.py --img0_path "../../data_sample/transformed_view.png" --img1_path ../../data_sample/original_views/ --out_path ../../matching_output/ --long_dim0 800 --long_dim1 800
```

#### filtering

### Embbeded deformation graph

### NeRF

Again, just run `conda install --file src/NeRF/environment.yaml` to install the conda virtual environment.

Here we provide the code that defines the forward flow to bend rays for NeRF rendering in the transformed scene. The full code for training the original NeRF and rendering from the transformed NeRF is coming soon. 

Our code is based on [ASpanFormer](https://github.com/apple/ml-aspanformer), [Nonrigid-ICP](https://github.com/rabbityl/Nonrigid-ICP-Pytorch) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) respectively. Thanks a lot to these great codebases!


CUDA_VISIBLE_DEVICES=7 python demo.py --img0_path "../data_nerf/cmc-render-main/dra_17/000009.png" --img1_path ../renders_high/dra_20_original_high/ --out_path /data01/zt15/outputs/aspan_outputs/output/dra_20_17_9_nerf_high --long_dim0 800 --long_dim1 800

CUDA_VISIBLE_DEVICES=7 python demo.py --img0_path "../data_sample/transformed_view/000009.png" --img1_path ../data_sample/original_views/ --out_path ../matching_output/ --long_dim0 800 --long_dim1 800