# KGNN-COMP683
Exploring Graph-Based Approaches for Predicting Drug Response from Kinome Inhibition States

## Main Env
```
mamba create -n kgnn torch-geometric networkx pandas numpy scikit-learn xgboost tqdm matplotlib pyreadr ipykernel pyarrow
```

## Node2Vec Env
(for graph_features.ipynb)
```
mamba create -n node2vec python=3.9 ipykernel pandas pyarrow matplotlib scikit-learn
mamba activate node2vec
pip install node2vec
```

## Files
* preproc: preprocess KIS and viability data. **filter viability data for samples that are between 0 and 1**
* graph_features: include node2vec embeddings with KIS data. embeddings are weighted by KIS so embeddings are not static. also includes viz of graph and pca on KIS vectors.
* baseline: train RF/XGB models using stratified 10-fold CV
* baseline_n2v: train RF/XGB using weighted n2v embeddings
* gnn: train baseline GNN
* gnn_feat: include learnable embeddings for cell and drug line
* evaluate: evaluate gnn models