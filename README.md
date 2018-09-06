# SegToolbox.PyTorch
We have implemented several semantic segmentation algorithms in PyTorch with comparable performance
Code will be updated in Oct.
## Model
- PSPNet
- DenseASPP
## Dataset
- CityScapes
- CoCo(TBD)
- ADE(TBD)
## Results

### CityScapes
|Algorithm|Model|Batch Size|Performance|Data|
|:------|:-------|:--------|:-------|:-------|
|DenseASPP|DenseASPP161|16|78.4%|Only fine|
|DenseASPP|DenseASPP209|-|-|Only fine|
|PSPNet|ResNet50|||Only fine|
|PSPNet|ResNet101|||Only fine|
|UPerNet|ResNet50|||Only fine|
|UPerNet|ResNet101|||Only fine|
