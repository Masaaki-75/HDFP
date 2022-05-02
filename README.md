# Holistic and Deep Feature Pyramids for Saliency Detection (HDFP)
This is a Pytorch re-implementation for a salient object detection model proprosed in the following paper:

[S. Dong, Z. Gao, et al., Holistic and Deep Feature Pyramids for Saliency Detection, British Machine Vision Conference, 2018.](http://bmvc2018.org/contents/papers/0212.pdf)

The code officially released is implemented in Tensorflow and is available [here](https://github.com/HIC-SYSU/HDFP). However, its [Pytorch version](https://github.com/zhifan-gao/HDFP-pytorch) seems not to precisely reflects the architecture of the HDFP model discussed in the paper, particularly in the part of building densely-connected layers (2022/05/02). So I try to refractor the code. 

Both new and old Pytorch versions of HDFP have only been tested on a private medical image dataset and the quantitative results are shown below. Detailed comparison on public salient object detection dataset will be added, hopefully, if necessary, in due course, at the appropriate juncture and in the fullness of time ;)
![mainimg](https://github.com/Masaaki-75/HDFP/blob/main/comp.png)


# Citation
```
@inproceedings{dong2018holistic,
  title={Holistic and Deep Feature Pyramids for Saliency Detection},
  author={Dong, Shizhong and Gao, Zhifan and Sun, Shanhui and Wang, Xin and Li, Ming and Zhang, Heye and Yang, Guang and Liu, Huafeng and Li, Shuo and others},
  booktitle={British Machine Vision Conference},
  year={2018}
}
```

# Environment
```
Python == 3.7
Pytorch == 1.11.0
```
