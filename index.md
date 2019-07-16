## Introduction

This is an ongoing effort for predicting human attention over webpages. Currently, the project contains a deep saliency baseline model based on ResNet architecture. New models will be added to the repository gradually over time.

### ResNet Baseline

It is based on ResNet-50 for encoding images and a decoder consisting of severeal blocks of convolution transpose and convolutions.


### Loss functions

We follow the findings of 



@inproceedings{Bruckert2019,
      title = {Deep Saliency Models : The Quest For The Loss Function,
     author = {Alexandre Bruckert and Hamed R. Tavakoli and Zhi Liu and Marc Christie and Olivier Le Meur},
       year = {2019},
  booktitle = {https://arxiv.org/abs/1907.02336}
}



