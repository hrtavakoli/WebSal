## Introduction

This is an ongoing effort for predicting human attention over webpages. Currently, the project contains a deep saliency baseline model based on ResNet architecture. New models will be added to the repository gradually over time.

### ResNet Baseline

It is based on ResNet-50 for encoding images and a decoder consisting of severeal blocks of convolution transpose and convolutions.


### Loss functions

We follow the findings of [Bruckert et al.](https://arxiv.org/abs/1907.02336) and employ a loss function which is a linear combination of three terms. The terms are based on KL-divergence, Correlation Coefficient, and Normalized Scan Path. Nevertheless, contrary to previous research, we adjust these terms to ensure not entering negative range values as follows:

$$ L = KLD + (1-\rho) + exp(-NSS) $$






