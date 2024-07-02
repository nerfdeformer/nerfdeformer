# NeRFDeformer

## Poster [Video](https://www.youtube.com/watch?v=oZsA6i9g_yM&t=1s)

## Data

please check [data](https://github.com/nerfdeformer/nerfdeformer/blob/main/data) for more datails.

## our evaluations

we put our full new view synthesis and geometry results in [our_evaluations](https://github.com/nerfdeformer/nerfdeformer/blob/main/our_evaluations).

## code

The code is separated to three parts: dense correspondence **matching**, **embedded deformation graph** (EDG) optimization, NeRF **new view synthesis in transformed scene** using EDG as deformation field. We have a data sample for inference unzip `src/data_sample.zip` to `src/data_sample` before running our code.

They are located in [src/correspondence_matching](https://github.com/nerfdeformer/nerfdeformer/blob/main/src/correspondence_matching), [src/EDG](https://github.com/nerfdeformer/nerfdeformer/blob/main/src/EDG) and [src/NeRF](https://github.com/nerfdeformer/nerfdeformer/blob/main/src/NeRF).

### correspondence matching

Our 2D corrspondence matching code is based on [ASpanFormer](https://github.com/apple/ml-aspanformer). We also use its outdoor pretrained model for inference. We include the pretrained model in the repository. 

To install the conda virtual environment, run `conda install --file src/correspondence_matching/environment.yaml`.

Then run the following commands to do matching on the sample data.

```

conda activate matching

cd src/correspondence_matching/demo

python match.py --img0_path ../../data_sample/transformed_view/transformed_view.png --img1_path ../../data_sample/original_views/ --out_path ../../matching_output/ --long_dim0 800 --long_dim1 800
```

Then you can check the 2D matching results in `src/matching_output`.

A clean code for two step filtering will come out in the next two weeks! (we have the uncleaned version in `src/correspondence_matching/filter')

### Embbeded deformation graph

The EDG optimization's code is based on [Nonrigid-ICP](https://github.com/rabbityl/Nonrigid-ICP-Pytorch). 

Run `conda install --file src/EDG/environment.yaml` to install the conda virtual environment.

A clean code for EDG will come out in the next two weeks! (we have the uncleaned version in `src/EDG')

### NeRF

Run `conda install --file src/NeRF/environment.yaml` to install the conda virtual environment.

Here we provide the code that defines the forward flow to bend rays for NeRF rendering in the transformed scene. The full code for training the original NeRF and rendering from the transformed NeRF is coming soon. 

## Acknowledgement

Our code is based on [ASpanFormer](https://github.com/apple/ml-aspanformer), [Nonrigid-ICP](https://github.com/rabbityl/Nonrigid-ICP-Pytorch) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) respectively. Thanks a lot to these great codebases!

## Citation
If you find our work useful in your project, please cite the following:

```
@inproceedings{tang2024nerfdeformer,
  title={NeRFDeformer: NeRF Transformation from a Single View via 3D Scene Flows},
  author={Tang, Zhenggang and Ren, Zhongzheng and Zhao, Xiaoming and Wen, Bowen and Tremblay, Jonathan and Birchfield, Stan and Schwing, Alexander},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10293--10303},
  year={2024}
}
```
