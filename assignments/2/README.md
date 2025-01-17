# Assignment 2 Report

## 3.2
- **Kkmeans1 = 3** from elbow plot( in figures/wcss_vs_k.png), as there is sudden decrease in wcss after 2 to 3, and a constant decrease from 3

## 4.2

- GMM (Gaussian Mixture Model) clustering works successfully in both the GMM class implementation and the sklearn implementation. Because various optimization techniques have been applied in both models.

  ### GMM Class
    - **Log-Sum-Exp Trick**: This is used to prevent overflow in log-likelihood calculations.
    - **Regularization**: A small regularization term (on the order of **1e-6**) is added to the covariance matrices to make them invertible.
    - **Small Float Addition**: Tiny float values are added to the log-likelihood to prevent errors such as taking the log of zero (**log(0)**).

  ### Sklearn
    - Similar to the custom implementation, sklearn also incorporates various optimization techniques to ensure stable and accurate clustering.

- **Kgmm1 = 1** from BIC and AIC values in both sklearn and GMM class

## 5.3
- In PCA, the new axes (principal components) represent directions of maximum variance in the data. Each component is a linear combination of original variables, capturing independent patterns. PC1 has the highest variance, with each subsequent component capturing decreasing amounts of remaining variance.

- from 2d(figures/pca_2d.png),and 3d(figures/pca_3d.png) plots of PCAs, we can say **K2 is around 3**


## 6
- from scree plot(figures/scree_plot.png) we can say that **optimal number of dimensions = 6**(as it is elbow point)
- using elbow, method optimal number of clusters in reduced dataset for k-means, **Kkmeans3 = 12**
- using AIC, BIC, optimal number of clusters in reduced dataset for GMM, **Kgmm3 = 1**

## 7

  ### 7.1

  - Clusters for Kkmeans1(=3) & K2(=3): 
  
    Cluster 0: ['hollow', 'saturn', 'badminton', 'earth', 'lantern', 'paintbrush']

    Cluster 1: ['drive', 'sing', 'listen', 'dive', 'flame', 'sit', 'knock', 'exit', 'brick', 'smile', 'bullet', 'bury', 'download', 'eat', 'postcard', 'hard', 'bend', 'fight', 'call', 'fly', 'face', 'climb', 'kneel', 'scream', 'kiss', 'passport', 'selfie', 'catch', 'hit', 'paint', 'far', 'dig', 'cry', 'notebook', 'run', 'clap', 'pull', 'sleep', 'baseball', 'puppy', 'clean', 'basket', 'sad', 'empty', 'fish', 'slide', 'drink', 'draw', 'pray', 'arrest', 'email', 'buy', 'burn', 'fire', 'close', 'angry', 'lazy', 'scary', 'hang', 'skate', 'hammer', 'book', 'tattoo', 'fingerprints', 'dustbin', 'tank', 'enter', 'key', 'swim', 'zip', 'happy', 'loud', 'love', 'knife', 'cook', 'basketball', 'arrow', 'recycle', 'cut', 'walk', 'sunny', 'radio', 'truck']

    Cluster 2: ['deer', 'panda', 'ape', 'rose', 'helicopter', 'cat', 'needle', 'eraser', 'table', 'carrot', 'fishing', 'bear', 'spider', 'shark', 'grass', 'giraffe', 'forest', 'lizard', 'brush', 'mug', 'feather', 'spoon', 'frog', 'puppet', 'gym', 'lake', 'monkey', 'roof', 'stairs', 'rifle', 'cow', 'pencil', 'bed', 'starfish', 'plant', 'microwave', 'knit', 'van', 'sweater', 'cigarette', 'microphone', 'jacket', 'bench', 'sun', 'bucket', 'feet', 'boat', 'pear', 'peacock', 'flute', 'fruit', 'scissor', 'grape', 'laptop', 'door', 'calendar', 'chair', 'mouse', 'ladder', 'finger', 'candle', 'ant', 'igloo', 'goldfish', 'bird', 'clock', 'oven', 'calculator', 'spiderman', 'bee', 'pillow', 'tree', 'beetle', 'envelope', 'toothbrush', 'screwdriver', 'snake', 'teaspoon', 'length', 'rain', 'airplane', 'ambulance', 'pizza', 'television', 'throne', 'tent', 'camera', 'tomato', 'dragonfly', 'parachute', 'butterfly', 'car', 'sword', 'loudspeaker', 'telephone', 'elephant', 'pant', 'stove', 'rainy', 'toothpaste', 'wheel', 'bicycle', 'windmill', 'toaster', 'potato', 'comb', 'crocodile', 'shoe', 'keyboard', 'fork', 'suitcase']

  - Clusters for Kkmeans3(=12):
    Cluster 0: ['needle', 'carrot', 'bend', 'puppet', 'scissor', 'finger', 'hammer', 'fingerprints', 'length', 'zip', 'sword', 'knife', 'arrow']

    Cluster 1: ['table', 'exit', 'climb', 'kneel', 'roof', 'bucket', 'feet', 'slide', 'door', 'ladder', 'clock', 'throne', 'wheel']

    Cluster 2: ['deer', 'bear', 'spider', 'shark', 'giraffe', 'lizard', 'frog', 'monkey', 'puppy', 'pear', 'grape', 'mouse', 'goldfish', 'bird', 'spiderman', 'beetle', 'snake', 'airplane', 'dragonfly', 'butterfly']

    Cluster 3: ['stairs', 'bed', 'sweater', 'jacket', 'bench', 'chair', 'igloo', 'tent', 'lantern', 'windmill']

    Cluster 4: ['listen', 'flame', 'knock', 'bury', 'download', 'hard', 'fight', 'call', 'hit', 'far', 'cry', 'sleep', 'clean', 'draw', 'pray', 'arrest', 'buy', 'burn', 'fire', 'close', 'angry', 'lazy', 'scary', 'hang', 'enter', 'happy', 'loud', 'love', 'cut']

    Cluster 5: ['panda', 'cow', 'starfish', 'peacock', 'ant', 'bee', 'tomato', 'elephant', 'crocodile']

    Cluster 6: ['brick', 'bullet', 'postcard', 'face', 'scream', 'passport', 'notebook', 'cigarette', 'hollow', 'empty', 'drink', 'email', 'book', 'dustbin', 'tank', 'pizza', 'key', 'cook', 'recycle', 'shoe', 'truck']

    Cluster 7: ['eraser', 'brush', 'feather', 'spoon', 'knit', 'flute', 'candle', 'toothbrush', 'screwdriver', 'teaspoon', 'toothpaste', 'comb', 'fork', 'paintbrush']

    Cluster 8: ['ape', 'sit', 'cat', 'eat', 'rifle', 'pencil', 'dig', 'run', 'clap', 'pull', 'van', 'sun', 'sad', 'camera', 'car', 'pant', 'potato']

    Cluster 9: ['drive', 'sing', 'rose', 'dive', 'fishing', 'smile', 'forest', 'fly', 'lake', 'kiss', 'selfie', 'catch', 'paint', 'plant', 'baseball', 'basket', 'fish', 'skate', 'tattoo', 'earth', 'rain', 'swim', 'rainy', 'basketball', 'walk', 'sunny']

    Cluster 10: ['mug', 'microwave', 'microphone', 'laptop', 'calendar', 'oven', 'calculator', 'pillow', 'envelope', 'ambulance', 'television', 'loudspeaker', 'telephone', 'stove', 'toaster', 'keyboard', 'radio', 'suitcase']

    Cluster 11: ['helicopter', 'grass', 'gym', 'boat', 'saturn', 'fruit', 'badminton', 'tree', 'parachute', 'bicycle'] 


  - The k=12 clustering approach yields more coherent and focused clusters than k=3. Specifically, k=12 clusters such as those grouping animals and related items (Cluster 2), furniture and shelter (Cluster 3), and animals with a food item (Cluster 5), show clearer coherence. Overall, k=12 provides more interpretable results. Therefore **Kkmeans = 12**

  ### 7.2

  - Kgmm1 and Kgmm3 = 1, all the words come into single cluster(i.e there is only 1 cluster)

  - k2 = 3

    Cluster 0: ['deer', 'ape', 'exit', 'brick', 'fishing', 'smile', 'spider', 'bullet', 'giraffe', 'forest', 'brush', 'feather', 'eat', 'spoon', 'puppet', 'call', 'fly', 'face', 'climb', 'scream', 'monkey', 'kiss', 'rifle', 'hit', 'bed', 'paint', 'microwave', 'cry', 'notebook', 'clap', 'knit', 'van', 'microphone', 'baseball', 'jacket', 'clean', 'peacock', 'saturn', 'flute', 'scissor', 'grape', 'badminton', 'ladder', 'pray', 'arrest', 'finger', 'goldfish', 'clock', 'spiderman', 'bee', 'hammer', 'book', 'tattoo', 'earth', 'dustbin', 'ambulance', 'television', 'key', 'tomato', 'dragonfly', 'butterfly', 'loudspeaker', 'elephant', 'knife', 'cook', 'toothpaste', 'basketball', 'bicycle', 'recycle', 'radio', 'truck', 'paintbrush']

    Cluster 1: ['drive', 'sing', 'rose', 'dive', 'sit', 'needle', 'eraser', 'table', 'carrot', 'bear', 'grass', 'bury', 'mug', 'bend', 'gym', 'lake', 'passport', 'roof', 'stairs', 'catch', 'pencil', 'starfish', 'run', 'pull', 'sleep', 'cigarette', 'bench', 'sun', 'puppy', 'pear', 'basket', 'sad', 'fish', 'drink', 'laptop', 'draw', 'calendar', 'chair', 'email', 'candle', 'igloo', 'oven', 'fire', 'scary', 'hang', 'envelope', 'fingerprints', 'teaspoon', 'length', 'rain', 'tank', 'airplane', 'pizza', 'enter', 'car', 'telephone', 'love', 'stove', 'rainy', 'arrow', 'toaster', 'comb', 'cut', 'shoe', 'walk', 'keyboard', 'sunny', 'suitcase']

    Cluster 2: ['panda', 'listen', 'flame', 'helicopter', 'knock', 'cat', 'shark', 'download', 'lizard', 'postcard', 'hard', 'frog', 'fight', 'kneel', 'selfie', 'cow', 'plant', 'far', 'dig', 'sweater', 'hollow', 'bucket', 'feet', 'boat', 'empty', 'slide', 'fruit', 'door', 'mouse', 'ant', 'buy', 'bird', 'calculator', 'burn', 'pillow', 'close', 'angry', 'lazy', 'tree', 'beetle', 'skate', 'toothbrush', 'screwdriver', 'snake', 'throne', 'swim', 'tent', 'camera', 'zip', 'parachute', 'sword', 'happy', 'lantern', 'loud', 'pant', 'wheel', 'windmill', 'potato', 'crocodile', 'fork']

  - The k=3 clustering approach for GMM provides better coherence and interpretability than k=1. While clusters in k=3 still vary, they offer clearer thematic groupings, especially in Cluster 1, which focuses on daily activities and objects. The k=1 clustering is too broad and lacks focus. Therefore **Kgmm = 3**

  ### 7.3

  - K-means (k=12):
    Has better coherence in several clusters (e.g., tools, animals, daily activities). Each cluster generally shows a clear theme, but some clusters remain broad, affecting overall interpretability.

  - GMM (k=3):
    Has fewer clusters with broader themes. Some clusters show better coherence around daily life and objects, but are less focused, and consistent within clusters due to the broader range of items and activities.

  - Kkmeans(=12) results in more and interpretable clusters, with better coherence in many cases. Kgmm(=3) provides a more general    view but lacks the granularity and specificity found in the K-means clusters. Therefore, Kkmeans(=12) is more effective for producing coherent and well-defined groupings.

## 8
- There is some overlap between labels of hierarchical clustering (complete linkage) and K-Means. However, hierarchical clustering tends to form fewer, larger clusters, grouping multiple points together, while K-Means creates smaller, more evenly distributed clusters. K-Means often splits the data into finer groups, whereas hierarchical clustering focuses on maximizing the distance between clusters, leading to larger, and cohesive groups.

- There is some overlap between labels of hierarchical clustering (using complete linkage) and GMM. However, GMM handles overlapping, flexible clusters due to its probabilistic nature, while hierarchical clustering (complete linkage) creates more compact and separate clusters with hard partitions. GMM provides a view with soft assignments, whereas hierarchical clustering forms well-separated, larger clusters by maximizing between-cluster distances.

## 9
- from scree plot(figures/scree_plot.png) we can say that **optimal number of dimensions = 2**(as it is elbow point)
- inference times for reduced data is less than actual data, as there are less number of features(or dimensions) in reduced data, and time complexity of knn is proportional to dimensions.
- values of metrics are low in reduced data, when compared to actual data(assingment-1 results), this indicates the loss of data.
