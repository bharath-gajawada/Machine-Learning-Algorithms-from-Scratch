# Report

## 2.3 GMM vs KDE

- **2 Components GMM**: The GMM with 2 components captures the general location of both clusters but struggles to model the circular shape of the large cluster.
- **Increasing Components**: Increasing the number of components allows the GMM to fit the data better, but it requires many components (each acting like a local Gaussian) to approximate a non-elliptical shape.
- **KDE**: KDE consistently captures the dataset structure across regions without needing to adjust the number of components, making it more flexible than GMM.

## 3.2 Observing MFCC Patterns

### MFCC Visualization
When visualizing the MFCC, you’ll see distinct temporal patterns. These patterns represent the unique transitions in spoken digits, which vary between speakers but still follow recognizable structures.

### Why HMM is Suitable
- **Sequential Data**: HMMs are ideal for sequential data, like spoken digits, where the probability of each state depends on the previous one.
- **Time-Varying Features**: MFCCs change over time as speech progresses, and HMMs can capture these dynamic changes effectively.


## 3.4 Model Performance

The model achieves **90.67%** accuracy on the test set, showing strong performance on familiar data. However, performance drops to **20.00%** on user-recorded data, indicating difficulties in generalizing to new speakers and recording conditions.


