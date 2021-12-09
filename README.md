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
- -![2731638237371_ pic](https://user-images.githubusercontent.com/44655900/145313981-70d2f82b-3c5f-4e2d-886a-5069f3b7002e.jpg)

- We also visualized the embeddings by using t-SNE to reduce the high-dimensional embeddings into two dimensions. Compared to the randomized embeddings initialized in our Word2Vec, we can clearly see that our final embeddings show four clusters of nodes, where each cluster represents a single category. This shows that DeepWalk is capable of encoding unstructured graphs into something useful and tangible for further graph analytics.

- Initial Radomized Embeddings before training
- ![Untrained skipgram visualization](https://user-images.githubusercontent.com/44655900/145314055-c6bb6f18-86fe-4e71-83c6-a28eb53165d0.png)

- Final embeddings after training 
- ![Figure_1](https://user-images.githubusercontent.com/44655900/145314039-b1bfbfec-3070-4240-b803-6a3e5199c771.png)
<!-- ## Contribution
Random Walk algorithm: Hangyu Du
Word2Vec: Yiwen Chen
Teaser + Poster:  -->
