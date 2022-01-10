## About
This repository contains the code of a PaddlePaddle2.2 implementation of STGCN based on the paper Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting https://arxiv.org/abs/1709.04875, with a few modifications in the model architecture to tackle with traffic jam forecasting problems.    

Forecasting traffic jams is not that similar to forecasting traffic flow, which we have a redundant amount of data. However, jams only occur in the largest cities and often only during peak hours, resulting in a unbalanced dataset. In order to study the jam patterns, we select roads in Haidian District, Beijing that have much more jam than others. 

## Related Papers
Semi-Supervised Classification with Graph Convolutional Networks https://arxiv.org/abs/1609.02907 (GCN)  

Inductive Representation Learning on Large Graphs https://arxiv.org/abs/1706.02216 （GraphSAGE）  

Graph Attention Networks https://arxiv.org/abs/1710.10903 （GAT）  

Bag of Tricks for Node Classification with Graph Neural Networks https://arxiv.org/pdf/2103.13355.pdf (BoT)  

Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting  https://ojs.aaai.org//index.php/AAAI/article/view/3881 (ASTGCN)

## Structural Modifications 
#### Graph operation  

The original STGCN model facilitates 1-st order ChebyConv and GCN as the graph operation. In our model we conducted experiments on one spectral method(GCN) and two spatial methods(GAT, GraphSAGE) 
#### Residual connection in graph convolution layer
Graph Neural Networks often suffer from oversmoothing: as the model becomes deeper, the representations of nodes tend to become similar to each other due to being repeatedly aggregated. Adding a residual connection mitigates oversmoothing by adding the input unsmoothed features directly to the output of graph convolution operation. Furthermore, the connection helps against gradient instablities.   

<img width="276" src="https://user-images.githubusercontent.com/20365304/144980066-f5936af9-961a-4f51-857a-269b35b3ffaa.png">

#### Incorporation of historical jam pattern
Jam status often follow daily patterns. In order to let the model learn historical patterns, we directly feed the model historical jam data with the same hour aligned. For example, if we want to predict the traffic status at 8PM. 30, Nov, 2021, we feed the model the 8PM traffic status in the past 12 days directly through a graph convolution layer, then concat it with the output of the S-T convolution blocks to generate the input of the final classifying layer.  

<img width="551" src="https://user-images.githubusercontent.com/20365304/144978158-b4baf9fd-a18c-40c5-9c77-dd73572f6ed3.png">

#### Multi-Step Prediction
The model is implemented to predict the jam status of several future time steps. First we feed the input data into the model to generate prediction of the first future time step. Then we concat the predicted status with the original input, feed to the model to generate the prediction of the next time step and so on. 

#### Classification
The original STGCN model was a regression model, optimizing a mean squared loss. Our traffic jam status has four classes: 1 -- smooth traffic; 2 -- temperate jam; 3 -- moderate jam; 4 -- heavy jam. So we changed it into a softmax with cross entropy classification model. Because in most of the cases, the traffic are smooth which makes label 1 dominates the others. We use a weighted cross entropy loss to punish incorrect classifications of 2, 3 and 4 more serverely. 

## Requirements
You can use pip to install the requirements:
```
sh requirements.sh
``` 

## Experiments


Right now I am still updating the experiments, adding new blocks, trying out new ideas.   
Here's a example of what a training epoch should look like right now, the numbers is the cross entropy loss of the model.  

<img width="526" alt="training" src="https://user-images.githubusercontent.com/20365304/148527350-afc54aa7-4ab0-4f6d-bd77-db69a3adbb64.png">


