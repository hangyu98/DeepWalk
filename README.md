# Deepwalk
## Introduction
We are implementing an existing paper called DeepWalk. DeepWalk aims to  learn social representations of a graph’s vertices by modeling a stream of random walks. Here, social representation refers to features of the vertices that capture the neighborhood similarities and communities. The resulting outcome of the model is low-dimensional embeddings for each node. Word embeddings are an important tool as they can be used in real-world applications, performing multiple tasks such as node classification, link prediction, community detections, etc. 

## Usage
#### <b>Note that all global vars and hyperparameters are in</b> ```./config.py```
#### Install all dependencies:
```pip install requirements```
#### Run DeepWalk:
```python deepwalk.py```
#### Visualize embeddings (optional):
```python visualize.py```

#### The enable gensim library implementation of Word2Vec(accelerated):
change variable  ```usingGensim``` to 'True' in ```config.py```(This is default setting)

Otherwise set ```usingGensim``` to 'True' to see our original implementation (takes longer to run)

## Evaluation
Node classification & Visualization
- After obtaining the embeddings for each node in the graph, we performed a multi-class node classification task to test the performance of our model. We trained a MLP classifier with a single hidden layer to classify embeddings into the predefined four categories. 80% of data are used for training, and 20% of the data are reserved for the test set. We achieved a weighted average of 92%
![image](https://user-images.githubusercontent.com/44655900/145313841-5d77675c-5caf-4973-9bea-25235944d267.png)
- We also visualized the embeddings by using t-SNE to reduce the high-dimensional embeddings into two dimensions. Compared to the randomized embeddings initialized in our Word2Vec, we can clearly see that our final embeddings show four clusters of nodes, where each cluster represents a single category. This shows that DeepWalk is capable of encoding unstructured graphs into something useful and tangible for further graph analytics.
![image](https://user-images.githubusercontent.com/44655900/145313852-987397a8-7953-4c86-9c6b-fad39a2c6ace.png)
![image](https://user-images.githubusercontent.com/44655900/145313859-11ffcce5-877e-4d08-9084-65e64ee9d722.png)


## Contribution
...
