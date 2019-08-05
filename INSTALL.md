# Installation

Requirements:
- PyTorch > 1.1.0
- torchvision
- yacs

## Step-by-step installation

```bash
conda create --name plusseg
conda activate plusseg

pip install yacs cython tqdm

# ninja yacs cython matplotlib tqdm opencv-python

pip install tensorboardX

# install apex
# use the commit 37795aac0d581918ccc33dc64c6480df74b82985
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

```