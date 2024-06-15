# NeRFDeformer

## Data

please check [data](https://github.com/nerfdeformer/nerfdeformer/blob/main/data) for more datails.

## our evaluations

we put our full new view synthesis and geometry results in [our_evaluations](https://github.com/nerfdeformer/nerfdeformer/blob/main/our_evaluations).

## code

The code is separated to three parts: dense **correspondence matching**, **embedded deformation graph** (EDG) optimization, **NeRF new view synthesis** in transformed scene using EDG as deformation field.

They are located in [src/correspondence_matching](), [src/EDG]() and [src/NeRF]().

### correspondence matching

#### filtering

### Embbeded deformation graph

### NeRF

the full code for training the original NeRF and rendering from the transformed NeRF is coming soon. We are 

Our code is based on [ASpanFormer](https://github.com/apple/ml-aspanformer), [Nonrigid-ICP](https://github.com/rabbityl/Nonrigid-ICP-Pytorch) and [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) respectively. Thanks a lot to these great codebases!