## Key Words
learning rate: how fast te model adapt to changing data
for high learning rate - the systems will rapidly adapt to new data but it  will tend to forget quickly old data
for low learning rate - the systems learns slowly and have more inertia but less sentitive to noise in the new data
instance based learning and model based learning:


## Use cases for Machine learning
* Problems for which existing solutions require alot of hand-turning or long list of rules.
* Complex problems for which there are no good solution at all using a traditonal approach.
* Fluating environments: Machine learning sytems can adapt to a new data
* Getting Insights about complex problems and large amounts of data

Types of supervise learning algorithm
* KNN
* Linear Regression
* Logistic Regression
* SVM
* Decison trees and random forest
* Neural networks

Types of unsupervised learning
* Clustering:
    * K-means
    * Hierachical clustering analysis (HCA)
    * dbscann
    * Expectation Maximization

* Visualization and Dimensionality redunction
    * PCA
    * Kernal PCA
    * Locally- linear Embedding (LLE)
    * t-distributed stochastic Neigbour Embedding (tsne)

* Association Mining
    * Apriori
    * Eclat


## Notes: it is often good to reduce the dimensionality of your data:
Before feeding it to another machine learning
### Hyperparameter controls the amounts or regularization to apply during learing not of the model but the learing algorithm


## Incremental learning and Batch Learning
Based on how
incremental learning also called online great for systems that recieves data in a continous flow and need to adapt to change rapidly.

## Instance based learning and Model based Learning
This is based on how the model generalize to unseen data

## Main challenges of Machine learning
### Data Challenges
* Insufficient Quantity of Training Data
* Non representative training Data: watc out for sample bias.
* Poor Quality data- ask how the data take is taken, what instrument, the state of the equipment. compare wit standard
* Irrelevant features - Machine learnig rely on feature engineering which involves feature selection, feature extracion, creating new feature by gathering new data

### Model challenges
* Overfitting the training data: when the model performs well on training data but does not generalize on new and unseen data.
it occurs when the model is too complex relative to te amount of noiseness of the training data.
* Underfitting the training data: occurs when the model is too simple to learn the underlying structure of the data

### In Ml project
* You gather data in the training set
* feed the training set to the learning algorithm
* for model based algo - it tunes some parameters to fit the model to te
