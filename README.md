## About
This repository contains the code of a PaddlePaddle implementation of STGCN based on the paper Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting https://arxiv.org/abs/1709.04875, with a few modifications in the model architecture to tackle with traffic jam forecasting problems.

## Related Papers
Semi-Supervised Classification with Graph Convolutional Networks https://arxiv.org/abs/1609.02907 (GCN)  

Inductive Representation Learning on Large Graphs https://arxiv.org/abs/1706.02216 （GraphSAGE）  

Graph Attention Networks https://arxiv.org/abs/1710.10903 （GAT）  

Bag of Tricks for Node Classification with Graph Neural Networks https://arxiv.org/pdf/2103.13355.pdf (BoT)  

Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting  https://ojs.aaai.org//index.php/AAAI/article/view/3881 (ASTGCN)

## Structural Modifications 
#### Residual connection in graph convolution layer
Graph Neural Networks often suffer from oversmoothing problems: as the layers become deep, the representations of node tend to become similar. Adding a residual connection mitigates the oversmoothing problem by adding the input unsmoothed features directly to the output of graph convolution operation. Furthermore, the connections helps against gradient instablities. 
<img width="276" alt="截屏2021-12-07 下午2 41 47" src="https://user-images.githubusercontent.com/20365304/144980066-f5936af9-961a-4f51-857a-269b35b3ffaa.png">

#### Incorporation of historical jam pattern
Jam status often follow weekly patterns. In order to let the model study historical patterns, we directly feed the model historical jam data with the same day and hour aligned. For example, if we want to predict the traffic status at 8PM. 30, Nov, 2021, we feed the model the 8PM traffic status in the past 12 tuesdays directly through a graph convolution layer. Then concat it with the output of the S-T convolution blocks to generate the input of the final classifying layer.
<img width="551" alt="截屏2021-12-01 下午3 35 25" src="https://user-images.githubusercontent.com/20365304/144978158-b4baf9fd-a18c-40c5-9c77-dd73572f6ed3.png">

#### Classification
The original STGCN model was a regression model, optimizing a mean squared loss. Our traffic jam status has four classes: 1 -- smooth traffic; 2 -- temperate jam; 3 -- moderate jam; 4 -- heavy jam. So we changed it into a softmax with cross entropy classification model.

## Requirements

## Experiments

## Further Directions 
