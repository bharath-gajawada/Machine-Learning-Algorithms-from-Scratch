# Assignment 1 Report

## 2.2.1

- **Correlation Heat Map**: 
  In the correlation heat map, we observe high correlation between the features `energy-loudness` (0.76) and `energy-acousticness` (-0.73). Due to this high correlation, these features have been removed from further analysis to reduce multicollinearity and improve model performance.

- **Duration Distribution Plot**: 
  The duration distribution plot reveals that the majority of the data points fall within the 0 to 1 minute range. There are very few data point beyond this range and can be considered as outliers, potentially skewing the analysis.

- **Speechiness vs Instrumentalness Plot**: 
  In the speechiness vs instrumentalness plot, most of the data points are concentrated towards the left end of the plot, represented by darker colors (violet), indicating lower liveliness. Additionally, there are a few scattered points outside of this range, which can be considered outliers.


## 2.5.1

- inference time is increased with training data set size, almost exponentially for both sklearn and best model.
- for small values of training data set size, sklearn model and best model take almost same inference time, but as training data set size is increased difference between inference times is significantly increased.

## 2.6

-Maybe the best hyper parameters for spotify.csv data set are not suited for this data set, resulting in relatively low metrics
-Data is likely to be noisy or randomized, because all the metrics are significantly low for the model.

## 3.2.1

-In L1 regularization as lambda value is increased predictions(y values) become almost same, and tend to 0 for very high values.
-In L2 regularization as lambda value is increased predictions(y values) come closer to each other (i.e graph becomes smoother).
