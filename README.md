# AutoTGRL

- AutoTGRL is an automatic text graph representation learning (AutoTGRL) framework, which can automatically achieve text graph representation learning for text classification tasks.

- The framework of AutoTGRL is as follows:

<br>
<div align=left> <img src="pic/AutoTGRL.SVG" height="100%" width="100%"/> </div>

<!-- ##  Instructions
The code has been tested with Python 3. To install the dependencies, please run:
```
pip install -r requirements.txt
``` -->
## Installing For Ubuntu 16.04

- **Ensure you have installed CUDA 10.2 before installing other packages**

**1. Nvidia and CUDA 10.2:**

```python
[Nvidia Driver] 
https://www.nvidia.cn/Download/index.aspx?lang=cn

[CUDA 10.2 Download and Install Command] 
#Download:
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#Install:
sudo sh cuda_10.2.89_440.33.01_linux.run

```

**2. Python environment:** recommending using Conda package manager to install

```python
conda create -n autotgrl python=3.7
source activate autotgrl
```

**3. Pytorch 1.8.1:** execute the following command in your conda env mvgnas

```python
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**4. Pytorch Geometric 2.0.2:** execute the following command in your conda env autotgrl
```python
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.2 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
```
**5. transformers:** execute the following command in your conda env autotgrl
```python
pip install transformers
```
**6. pytorch-ignite:** execute the following command in your conda env autotgrl
```python
pip install pytorch-ignite
```


## Running the Experiment
For cleaning text dataset, please refer to the script 'main_clean_data.py' 
```python
python main_clean_data.py --dataset R52 --model_type 'transductive'
```

For text graph representation, please refer to the script 'main_build_graph.py' 
```python
python main_build_graph.py --dataset R52 --model_type 'transductive'
```

For generating initial document embedding, please refer to the script 'finetune_bert.py' 
```python
python large_scale_embd/finetune_bert.py --dataset R52 --bert_lr 1e-4  --bert_init  'roberta-base'
```

For training, please refer to the script 'main.py' 
```python
python main.py --dataset R52 --model_type 'transductive'
```

Note: more details will be added soon.
