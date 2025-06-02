## How would you define Machine Learning
Machine learning entails building systems that learn from examples( data) to carry out a task. The perfomance of the system on the task increase as more examples are available. It a system that learns from data without being explicitly programme to carry some task.

is the science and art of programming computers so that they can learn from data.
it is a field of study that gives computers the ability to learn without being explicitly programmed

## Four types of problems where machine learning
* for problems that requires long list of rules or hand tunning
* getting insight large amounts of data
* fluacting environment
* complex problems that traditional ml can't solve.

## What is the labeled training set
a set of examples where both inputs and desired outputs are already paired.
A labeled training set (often called a “labeled dataset”) is simply a collection of examples where each example consists of:

Input features (often called x)

A corresponding “label” (often called y)

In other words, for every data point in the set, you already know the correct answer (the label). This is the core requirement for supervised learning.

## what are two most common supervised task
* Classification and Regression task

## Can you Name Four common Supervised tasks
* Clustering
* visualization and dimensionality Analysis
* Association mining
* Anomaly detection

## what type of  algorithm allows a robot to learns and walk in various unknown terrain
Reinforcement learning

## Algorithm to segments customer in to multiple groups
Clustering

## Spam detection problem
Spam detection problem should be frame as supervised learning

## What is an online learning system
This is also known as incremental learning where the data is chop into streams of mini batch and use to train the model and these is repeated for subsequent batches,

Batch Learning

Definition:
Batch learning (also called “offline learning”) means you train your model on a fixed dataset all at once. You gather all your training examples first, then run the learning algorithm over that entire dataset to build or update the model.

How It Works:

Collect Data: Assemble your full training set (e.g., 10 000 labeled examples).

Train Model: Feed all 10 000 examples into the learning algorithm. It processes them—often in multiple passes—until the model’s parameters converge (e.g., weights in a neural network, split thresholds in a decision tree).

Deploy: Once training finishes, you have a static model. To improve it later, you typically “retrain” from scratch on the entire dataset (old + new examples).

Key Characteristics:

Fixed Dataset: The model never sees new data until you explicitly retrain.

High Initial Computational Cost: Training can be expensive because the algorithm processes every example, sometimes many times.

Stable Predictions: Since the model only changes when you retrain, predictions remain consistent between training sessions.

Easy to Debug & Reproduce: You know exactly which data went into training and when.

Practical Example:

Suppose you have 50 000 images of handwritten digits labeled 0–9. You run a convolutional neural network (CNN) training procedure on all 50 000 images for, say, 20 epochs. Once those epochs finish, you save that trained network. If you later collect 5 000 more images, you’d combine them with the original 50 000 and retrain the CNN on the full 55 000 images to update it.

Online Learning

Definition:
Online learning (sometimes called “incremental learning” or “streaming learning”) means the model is updated continuously or incrementally as new data arrives—often one example (or a small batch) at a time—rather than waiting to see the entire dataset at once.

How It Works:

Initialize Model: Start with an initial (possibly random) model.

Receive One (or Small Batch) of Examples: As each new example arrives, you immediately update the model’s parameters based on that example alone (or a tiny mini-batch).

Repeat Continuously: The model “learns” on the fly. There isn’t a final “training” step—learning and prediction happen in a continuous loop.

Key Characteristics:

Data Stream: The model never needs to store all past data. It just processes each example once (or a few times) as it arrives.

Low Memory Footprint: Only current model parameters and perhaps a tiny buffer of recent examples are kept in memory—no huge dataset storage.

Fast Adaptation to Change: Because you update immediately on each new example, the model can quickly adapt if the underlying data distribution shifts.

Potentially Noisy Updates: Learning from one example at a time can be noisy, so algorithms often use a “decaying learning rate” or small buffers to stabilize updates.

Practical Example:

Imagine a spam filter for your email. Instead of collecting a week’s worth of emails and then retraining, the filter checks each incoming email’s features (sender, subject, keywords). After you mark one email “spam,” the filter updates its weights right away based on that single labeled example. The next email is processed by the slightly updated model. Over time, as more emails arrive, it keeps refining its parameters example by example.

Comparison

Aspect	Batch Learning	Online Learning
Data Availability	Needs the entire training set available upfront.	Works with a continuous stream of data, one at a time.
Memory Requirements	Must store (and revisit) all training data.	Only store current model + small buffers; not full dataset.
Training Time	Potentially long, since it processes the full dataset repeatedly.	Generally fast per example; updates happen incremental.
Adaptation to Changes	Slow: model stays the same until you retrain on new data.	Fast: instantly updates whenever new example arrives.
Use Cases	– When data is static (e.g., historical records).
– When you want a well‐tuned model offline.	– When data arrives continuously (e.g., real-time sensor readings).
– When you need to adapt quickly to shifting patterns.

When to Choose Which

Batch Learning:

Your data is all collected and labeled before training (e.g., historical sales records up to last month).

You can afford to retrain from scratch periodically (e.g., once a day or once a week).

You need stable, reproducible results and can wait for longer training runs.

Online Learning:

Your data flows in continuously (e.g., stock‐price ticks, user clicks, social media posts).

You want the model to adapt in real time or near‐real time.

You cannot store the full dataset (e.g., IoT devices with limited memory).

In a Nutshell
Batch Learning: “Wait until I have all my data, then train my model in one go (or a few passes).”

Online Learning: “Update my model bit by bit as each new example arrives, so it’s always up to date.”

## What is out of core learning-
Out-of-core learning refers to machine learning methods designed to handle datasets that are too large to fit into a computer’s RAM all at once. Instead of loading the entire dataset into memory, the algorithm reads and processes data in smaller chunks (batches) directly from disk (or another external source), updating its model incrementally as it goes.

1. Why Out-of-Core Learning Exists
Memory Limitations:
If you have, say, 100 GB of training data but only 16 GB of RAM, you cannot simply call “load all data” and train in one shot. Out-of-core learning solves this by streaming data in pieces.

Large-Scale Applications:
When working with web-scale logs, video frames, or massive image corpora, it’s common for the dataset to exceed RAM. Out-of-core techniques ensure you can still train models without requiring a supercomputer.

2. How It Works
Chunking Data on Disk:

The dataset lives in files (e.g., CSVs, TFRecords, databases).

Instead of reading everything at once, you read a manageable “chunk” or “mini-batch” (e.g., 10 000 rows) into memory.

Incremental Model Updates:

For each chunk, the algorithm updates its parameters (weights, splits, centroids, etc.).

Once you finish processing that chunk, you discard it from RAM (or write back any necessary summaries).

Repeat Until All Data Is Processed:

Continue reading the next chunk and updating the model until you’ve seen every example on disk.

At the end, you have a model that has “seen” the whole dataset, but at no point did you hold the entire dataset in RAM.

Optional Multiple Passes (Epochs):

Some out-of-core methods make just one pass through the data (one epoch).

Others allow several passes—simply rewind to the start of the file(s) and repeat chunked updates, if you need more training iterations.

3. Relationship to Batch vs. Online Learning
Aspect	Batch Learning	Online Learning	Out-of-Core Learning
Data Size	Fits entirely in memory	Can be infinite or streaming	Exceeds memory; stored on disk
Data Access Pattern	Read full dataset into RAM	Read one example (or tiny batch)	Read moderate‐sized chunks (e.g., few MB)
Model Updates	After loading all data, train	Update per example (or tiny batch)	Update per chunk (mini-batch on disk)
Memory Usage	High (data + model in RAM)	Low (single example at a time)	Moderate (one chunk at a time)
Use Cases	Static, relatively small datasets	Truly streaming environments	Large but finite datasets stored on disk

Batch Learning assumes the entire dataset fits in RAM.

Online Learning assumes data arrives continually (e.g., a live stream), and you update immediately per example.

Out-of-Core Learning is for a fixed—but too-large-for-memory—dataset: you process it chunk by chunk, downloading parts from disk, updating the model, then discarding that chunk.

4. Practical Examples
Scikit-Learn’s partial_fit Interface

Many classifiers/regressors in scikit-learn (e.g., SGDClassifier, MiniBatchKMeans) support a partial_fit method.

How to use it:

python
Copy
Edit
from sklearn.linear_model import SGDClassifier
import pandas as pd

model = SGDClassifier(loss='log')  # logistic regression via SGD

# Suppose “large_dataset.csv” is 50 GB on disk.
chunk_size = 10_000
for chunk in pd.read_csv("large_dataset.csv", chunksize=chunk_size):
    X_chunk = chunk.drop("label", axis=1).values
    y_chunk = chunk["label"].values
    model.partial_fit(X_chunk, y_chunk, classes=[0,1,2])  
    # classes=[…] needed on first call for multiclass.

# After looping through all chunks, `model` has learned from the entire 50 GB dataset.
Why It’s Out-of-Core: You never load more than 10 000 rows (plus whatever internals the model needs) at once.

TensorFlow’s tf.data Pipelines with from_generator or TFRecordDataset

You can store your training examples as TFRecord files on disk.

Then use a tf.data.Dataset that reads them in fixed-size batches, preprocesses them, and feeds them into a Keras model.

Because TensorFlow only keeps the current batch in memory, you can train on arbitrarily large datasets.

Vowpal Wabbit (VW)

An open-source tool specifically designed for extremely large, streaming datasets.

It reads plain text or binary data from disk line by line, updates its model (often a logistic or linear-regression variant) incrementally, and writes out a final model without ever requiring the full dataset in RAM.

Commonly used for click-through‐rate prediction in ad‐tech, where the data easily runs into hundreds of gigabytes.

Out-of-Core Clustering with Mini-Batch K-Means

The standard K-Means algorithm requires all data in memory to find the nearest centroid for each point.

MiniBatchKMeans (in scikit-learn) instead:

Samples a small batch of points (e.g., 1 000 at a time), updates centroids based on that batch, discards the batch, and repeats—eventually covering the entire dataset on disk.

This drastically reduces memory overhead, allowing clustering over data that would never fit in RAM.

5. When to Use Out-of-Core Learning
Dataset Size > RAM Capacity:
If your training files collectively are larger than your available memory (e.g., hundreds of GB vs. tens of GB RAM), out-of-core is the natural choice.

No Distributed Cluster Available:
If you can’t spin up a Spark or Hadoop cluster, but you do have a single machine with limited RAM, out-of-core algorithms let you train on local disk.

Faster Prototyping vs. Garbage Collection:
Instead of wrestling with CSV→database→SQL to sample smaller bits (which can be slow), you can directly stream from disk in code, iterate over chunks, and avoid complex data‐engineering steps.

Avoiding Complete Retraining:
Some out-of-core learners allow you to checkpoint model state to disk and later resume training—handy when you need multiple epochs over a giant dataset.

6. Key Trade-Offs
Pros:

Scalability: Train on arbitrarily large datasets without blowing out memory.

Simplicity for Single‐Machine Setups: No need for distributed infrastructure—just stream from local or network‐mounted disk.

Incremental Checkpointing: You can save partial model states and resume if training is interrupted.

Cons:

Longer I/O Time: Reading from disk in chunks is slower than pure in-memory operations.

Algorithmic Limitations: Not every algorithm supports out-of-core updates; many require random‐access to all data.

Model Quality vs. Mini-Batch Size: If your chunks are too small, gradient estimates (for SGD, for example) can be noisy, potentially slowing convergence.

Multiple Passes Are Costly: If you need many epochs, you’ll re-read the same files over and over from disk, which can be slow.

7. In a Nutshell
Out-of-Core Learning = training directly on data stored on disk by reading and processing it in smaller “chunks,” instead of requiring the full dataset in RAM.

It’s the bridge between “batch” (all data in memory) and “online” (one example at a time in a stream), optimized for large but finite datasets that simply cannot fit into memory.

Common tools/approaches include:

Scikit-learn’s partial_fit() methods (e.g., SGDClassifier, MiniBatchKMeans).

TensorFlow’s tf.data pipelines with TFRecord or generator functions.

Specialized systems like Vowpal Wabbit.

Use out-of-core when your data is too big for RAM, yet you want to train on a single machine without distributed computing.


## What time of type of learning algorithm relies on similarity measure to make predictions
instance based learning

## what is the difference between model paramete and hyperameter
Model Parameters are the internal variables that a learning algorithm adjusts automatically during training to fit the data. Once training finishes, the parameters encode the learned relationships.

Hyperparameters are external configuration settings whose values are set before (or during) training and control aspects of the learning process itself. The algorithm does not learn these from the data; you choose them (sometimes via search/validation).

1. Model Parameters
What they are:

Numerical values (weights, coefficients, biases, centroids, etc.) that define the specific form of the trained model.

They directly influence how the model transforms inputs into outputs.

How they’re found:

During training, the algorithm iteratively optimizes parameters to minimize a loss function on the training data.

Example in linear regression: if your hypothesis is

𝑦
^
=
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
𝑏
,
y
^
​
 =w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +b,
then 
𝑤
1
,
𝑤
2
,
w 
1
​
 ,w 
2
​
 , and 
𝑏
b are parameters that gradient descent (or another optimizer) adjusts so that predicted 
𝑦
^
y
^
​
  closely matches the true 
𝑦
y.

Examples:

Neural Network Weights & Biases

Each connection between neurons has a weight; each neuron often has a bias term. During backpropagation, these parameters update to reduce classification or regression error.

Decision Tree Split Thresholds

In a trained decision tree, each internal node’s “feature index” and “threshold value” (e.g., “feature — age > 30”) are parameters learned from data.

K-Means Centroids

The centroids’ coordinates (for each cluster) are parameters found by iteratively assigning points and recomputing cluster centers.

2. Hyperparameters
What they are:

Settings or “knobs” that govern how training proceeds or how complex the model can become.

They are not optimized directly by the model’s fitting algorithm; instead, you select (or search over) them externally, often via cross-validation or grid/random search.

How they’re chosen:

Before training begins, or via a nested search procedure: you pick a candidate set of hyperparameter values, train a model for each combination, evaluate performance on a validation set, and select the best.

Sometimes you adjust them manually based on experience or domain knowledge.

Common Examples:

Learning Rate (
𝛼
α) in Gradient Descent / Neural Nets

Controls the step size for updating parameters. Too large → divergence; too small → extremely slow convergence.

Number of Hidden Layers & Neurons (Neural Network Architecture)

Decides how “deep” or “wide” the network is before training. Affects representational capacity and overfitting risk.

Regularization Strength (e.g., 
𝜆
λ in Ridge/Lasso Regression)

Controls how much the model penalizes large weights. A larger 
𝜆
λ forces weights toward zero, reducing overfitting but possibly underfitting.

Number of Trees & Maximum Depth (Random Forest / Gradient-Boosted Trees)

“Number of trees = 100,” “max depth = 5,” etc. These shape model complexity and training duration.

Batch Size in Mini-Batch Gradient Descent

The number of training examples the algorithm uses before updating parameters. A small batch size adds noise to gradients; a large batch size uses more memory and may converge differently.

3. Key Differences
Aspect	Model Parameters	Hyperparameters
Learned From Data?	Yes—automatically optimized during training	No—set before (or via external search)
Role	Define the final function or decision rules	Control how training is performed or model structure
Examples	Weights, biases, tree split values, centroids	Learning rate, regularization coefficient, number of layers, batch size
Optimization	Learned by minimizing a loss (e.g., gradient descent, EM)	Chosen via cross-validation, grid/random search, Bayesian optimization
Adjusting Frequency	Continuously updated at each training step	Fixed for each training run (unless you implement dynamic schedules, e.g., learning‐rate decay)

4. Practical Illustration
Imagine you want to train a neural network to classify images of handwritten digits:

Set Hyperparameters (Before Training):

Number of hidden layers = 2

Neurons per layer = [128, 64]

Activation function = ReLU

Learning rate = 0.01

Batch size = 64

Number of epochs = 20

Regularization (dropout rate) = 0.5

During Training:

The algorithm repeatedly performs forward and backward passes over the data batches.

In each pass, model parameters—namely, the weights and biases of every neuron—are updated (e.g., weight := weight − learning_rate × ∂loss/∂weight).

After Training:

The final parameters encode precisely how to map pixel values of a new image to a digit.

The hyperparameters chosen beforehand (e.g., learning rate = 0.01, dropout rate = 0.5) remain fixed—they helped shape the training process but are not part of the final network weights.

5. Why This Distinction Matters
Performance Tuning:

Good hyperparameter choices (e.g., right learning rate, adequate regularization) are crucial for convergence speed and avoiding overfitting/underfitting.

Model parameters alone cannot compensate if hyperparameters are terribly set (e.g., learning rate too high causes divergence).

Reproducibility & Comparison:

When you report a trained model’s performance, you need to document both the final parameter values and the hyperparameter settings used. Two models with identical architectures but different hyperparameters can behave very differently.

Automated Hyperparameter Search:

Tools like grid search, random search, or Bayesian optimization specifically explore hyperparameter combinations—keeping model‐parameter optimization “inner loop” and hyperparameter‐selection “outer loop.”

In a Nutshell
Model Parameters = “What the model learned from your data.”

Hyperparameters = “The knobs you set beforehand to guide how the model learns.”

## Model based learning
What model-based learning algorithms search for: These algorithms search for an optimal value for the model parameters. The objective is to find parameter values such that the model will generalize well to new instances. When the algorithm is model-based, it tunes parameters to fit the model to the training set, making good predictions on the training data itself, with the hope that it will then be able to make good predictions on new cases.
•
What is the most common strategy they use to succeed: The usual strategy to succeed is to train these systems by minimizing a cost function. This cost function measures how bad the system is at making predictions on the training data. For linear regression problems, for example, a typical cost function measures the distance between the model's predictions and the training examples, and the objective is to minimize this distance. Additionally, this strategy can include adding a penalty for model complexity if the model is regularized. The learning algorithm is fed the training examples and finds the parameters that make the model fit best to the data, minimizing the cost function.
•
How they make predictions: To make predictions on new cases (called inference), you feed the new instance's features into the model's prediction function. This prediction uses the specific model parameter values that were found by the learning algorithm during training. For example, after training a linear model to predict life satisfaction based on GDP per capita and finding the optimal parameters, you can use these parameters in the linear function to predict life satisfaction for a new country based on its GDP per capita.

## What is overfitting and how do you solve it
It occurs when a model performs well on the training data, but it does not generalize well to new instances.
•
Overfitting can be likened to a human tendency to overgeneralize from limited experience, such as assuming all taxi drivers in a foreign country are thieves after one negative experience.
•
It often happens when the model is too complex relative to the amount and noisiness of the training data. A complex model can detect subtle patterns, but if the training set is noisy or too small, it may detect patterns in the noise itself, which will not generalize to new instances.
•
A model with many degrees of freedom (like a high-degree polynomial model or a deep neural network with millions of parameters) is likely to have high variance and thus be prone to overfitting. This is part of the bias/variance tradeoff.
•
You can tell a model is overfitting if its error rate on the training data is low, but its generalization error (or out-of-sample error) on new cases is high. The generalization error is typically estimated using a test set.
•
If you plot the model's performance on the training set and a validation set (learning curves), a large gap between the training error and the validation error is a hallmark of an overfitting model. If the validation error consistently goes up while the training error doesn't, the model is overfitting.
Addressing overfitting is crucial for building effective Machine Learning systems. The sources suggest several ways to solve or avoid overfitting:
1.
Get More Training Data: Feeding the model more training data can improve an overfitting model. Data augmentation, which involves generating new training instances from existing ones, can artificially boost the training set size and reduce overfitting.
2.
Simplify the Model:
◦
Select a simpler algorithm.
◦
Reduce the number of parameters in the model. For example, use a linear model instead of a high-degree polynomial model, reduce the polynomial degree, or for MLPs, reduce the number of hidden layers and neurons per layer.
◦
Reduce the number of attributes (features) in the training data.
3.
Regularize the Model: Regularization means constraining the model to make it simpler and reduce the risk of overfitting. The amount of regularization is controlled by a hyperparameter. Common regularization techniques include:
◦
Ridge Regression (ℓ2 regularization) and Lasso Regression (ℓ1 regularization) add a penalty for large weights to the cost function, forcing the learning algorithm to not only fit the data but also keep model weights small. For Ridge Regression, increasing the regularization hyperparameter α reduces overfitting. Lasso can lead to sparse models where some weights are exactly zero.
◦
Elastic Net combines ℓ1 and ℓ2 regularization.
◦
Early Stopping is a very effective technique for iterative algorithms (like Gradient Descent variants or training neural networks and Gradient Boosting ensembles) where you stop training as soon as the model's performance on the validation set starts to drop.
◦
For SVMs, reducing the hyperparameter C regularizes the model and reduces overfitting. For SVMs with an RBF kernel, reducing the gamma hyperparameter also acts like regularization.
◦
For Decision Trees, reducing the max_depth hyperparameter regularizes the model.
◦
For AdaBoost ensembles, reducing the number of estimators or regularizing the base estimator helps with overfitting.
◦
For Gradient Boosting ensembles, decreasing the learning rate or using early stopping to find the optimal number of trees helps with overfitting.
◦
For Deep Neural Networks, popular regularization techniques include:
▪
Early Stopping.
▪
ℓ1 and ℓ2 regularization applied to the weights.
▪
Dropout, where neurons are randomly turned off during training. If the model is overfitting, you can increase the dropout rate.
▪
Max-Norm Regularization, which constrains the maximum norm of the weight vectors. Reducing the constraint r increases regularization.
▪
Data Augmentation.
▪
Reducing the number of hidden layers and neurons per layer.
▪
Tying weights in autoencoders reduces parameters and overfitting risk.
4.
Reduce Noise in Training Data: Fixing data errors and removing outliers can help reduce overfitting.