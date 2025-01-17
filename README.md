# Machine Learning Algorithms from Scratch

A comprehensive collection of machine learning algorithms implemented from scratch in Python, focusing on fundamental concepts and hands-on implementation. This repository contains assignments covering various ML techniques, from basic classification to deep learning.

## 🎯 Core Features

- Pure Python/NumPy implementations of fundamental ML algorithms
- Extensive visualization and analysis tools
- Comprehensive performance metrics and comparisons
- Integration with Weights & Biases for experiment tracking
- Support for multiple datasets including MNIST, Fashion MNIST, and Spotify data

## 📚 Implementations

### Classical Machine Learning
- **KNN Classification**
  - Custom KNN implementation with multiple distance metrics
  - Hyperparameter tuning and optimization
  - Vectorized implementation for performance
  - Music genre prediction using Spotify dataset

- **Linear Regression**
  - Multi-degree polynomial fitting
  - L1 and L2 regularization
  - Animated visualization of fitting process
  - MSE, standard deviation, and variance analysis

- **Clustering Algorithms**
  - K-Means clustering with elbow method optimization
  - Gaussian Mixture Models (GMM) with BIC/AIC analysis
  - Hierarchical clustering with dendrograms
  - Word embeddings clustering analysis

### Neural Networks

- **Multi-Layer Perceptron**
  - Classification and regression implementations
  - Multiple activation functions (ReLU, Sigmoid, Tanh)
  - Various optimization techniques (SGD, Mini-batch, Batch)
  - Gradient checking and validation
  - Integration with W&B for hyperparameter tuning

- **Convolutional Neural Networks**
  - Multi-digit MNIST classification
  - Multi-label classification
  - Feature map visualization
  - Custom CNN implementations for both classification and regression

- **Autoencoders**
  - CNN-based autoencoder implementation
  - MLP-based autoencoder
  - PCA-based dimensionality reduction
  - Comparative analysis between approaches
  - Fashion MNIST reconstruction and classification

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**
  - Custom implementation with transformation capabilities
  - Optimal component selection
  - Visualization in 2D and 3D spaces
  - Performance comparison with autoencoders

## 🛠️ Dependencies
- NumPy
- Pandas
- Matplotlib
- PyTorch
- Scikit-learn (for comparison only)
- Weights & Biases (WandB)
- Librosa (for audio processing)

## 📊 Datasets
- Wine Quality Dataset
- Fashion MNIST
- Free Spoken Digit Dataset
- Spotify Music Features
- Multi-MNIST
- Custom synthetic datasets