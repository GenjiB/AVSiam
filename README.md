# Siamese Vision Transformers are Scalable Audio-visual Learners
[📗Paper](https://arxiv.org/abs/2403.19638)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="https://raw.githubusercontent.com/facebookresearch/unbiased-teacher/main/teaser/pytorch-logo-dark.png" width="10%"> 


This is the PyTorch implementation of our paper: <br>
### Siamese Vision Transformers are Scalable Audio-visual Learners <br>
[Yan-Bo Lin](https://genjib.github.io/) and [Gedas Bertasius](https://www.gedasbertasius.com/)<br>


<p align="center">
<img src="https://i.imgur.com/z6A5kGd.png" width="70%">
</p>

<br>


<br>**Our Method**<br>

<p align="center">
<img src="https://i.imgur.com/1gOUGh3.png" width="80%">
</p>

### 📝 Preparation 
*  `pip3 install -r requirement`
* Download AudioSet and VGGSound 
* Download [jx_vit_base_patch16_224_in21k-e5005f0a.pth](https://huggingface.co/genjib/AVSiam/blob/main/jx_vit_base_patch16_224_in21k-e5005f0a.pth) and save at `./src/adapt_weights` (Not necessary. But, it somehow affect results a bit.)
* Donwload [sqllite3 files](https://huggingface.co/genjib/AVSiam/tree/main/sql) and save wherever you want. (Instead of reading csv annotation, this can address out of CPU memory issue)
* edit `./scr/dataloader.py` and `./scr/dataloader_ft.py` to make sure your video path and sql path is correct.

### 🏃 Pretraining
* `run ./egs/audioset/run_pretrain_base.sh`

### 🏃 Finetuneing
* AudioSet 2M:`run ./egs/audioset/run_base_ft_2m.sh`
* AudioSet 20K:`run ./egs/audioset/run_base_ft.sh`
* VGGSound: `run ./egs/vggsound/run_base_ft.sh`

### 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@article{lin2024siamese,
  title={Siamese Vision Transformers are Scalable Audio-visual Learners},
  author={Lin, Yan-Bo and Bertasius, Gedas},
  journal={arXiv preprint arXiv:2403.19638},
  year={2024}
}
```

### 👍 Acknowledgments
Our code is based on [CAV-MAE](https://github.com/YuanGongND/cav-mae). 


### ✏ Model Checkpoints
More Checkpoints and training scripts will be available.

| Base  |  Base+  | Large     | Huge
|-------------|-----------------| ---------------| ---------------|
| [PT AS-2M](https://huggingface.co/genjib/AVSiam/blob/main/as2m_pretrained.20.pth)        | [PT AS-2M+VGG+ACAV2.4M](https://huggingface.co/genjib/AVSiam/blob/main/as%2Bvgg%2Bacav_pretrained.pth)      

