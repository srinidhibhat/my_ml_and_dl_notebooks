# Hands on ML Notes - Part 1 (The Fundamentals of Machine Learning)
## CHAPTER 1: The Machine Learning Landscape
- Applying ML techniques to dig into large amounts of data can help discover patterns that were not immediately apparent. This is called data mining. 
- In Machine Learning an attribute is a data type (e.g., “Mileage”), while a feature has several meanings depending on the context, but generally means an attribute plus its value (e.g., “Mileage = 15,000”). Many people use the words attribute and feature interchangeably, though.
- Dimensionality reduction is a task in which the goal is to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be very correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called 'feature extraction'.
- Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards). It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.
- In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.
- In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.
- One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data.
- One more way to categorize Machine Learning systems is by how they generalize.There are two main approaches to generalization: 
    1. Instance-based learning: Possibly the most trivial form of learning is simply to learn by heart. If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users—not the worst solution, but certainly not the best.
    2. Model-based learning: Another way to generalize from a set of  examples is to build a model of these examples, then use that model to make predictions. This is called model-based learning.
- How can you know which values will make your model perform best? To answer this question, you need to specify a performance measure. You can either define a utility function (or fitness function) that measures how good your model is, or you can define a cost function that measures how bad it is.
- Main Challenges of Machine Learning:
    - Insufficient Quantity of Training Data: Sometimes corpus development is more important than algorithm development.
    - Nonrepresentative Training Data: In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise, but even very large samples can be nonrepresentative if the sampling method is flawed. This is called 'sampling bias'.
    - Poor-Quality Data: Obviously, if your training data is full of errors, outliers, and noise, it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well.
    - Irrelevant Features: A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called 'feature engineering', involves: Feature selection, Feature extraction (combining existing features to produce a more useful one) and Creating new features by gathering new data.
    - Overfitting the Training Data: It means that the model performs well on the training data, but it does not generalize well. Constraining a model to make it simpler and reduce the risk of overfitting is called 'regularization'. The amount of regularization to apply during learning can be controlled by a 'hyperparameter'. A hyperparameter is a parameter of a learning algorithm (not of the model).
      - Underfitting the Training Data: It occurs when your model is too simple to learn the underlying structure of the data.

- Testing and Validating: You train multiple models with various  hyperparameters using the training set, you select the model and hyperparameters that perform best on the validation set, and when you’re  happy with your model you run a single final test against the test set to get an estimate of the generalization error. To avoid “wasting” too much training data in validation sets, a common technique is to use 'cross-validation': the training set is split into complementary subsets, and each model is trained against a different combination of these subsets and validated against the remaining parts. Once the model type and hyperparameters have been selected, a final model is trained using these hyperparameters on the full training set, and the generalized error is measured on the test set.

## CHAPTER 2: End-to-End Machine Learning Project

### Look at the Big Picture
- Pipelines: A sequence of data processing components is called a data pipeline. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply.
- Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values.  Various distance measures, or norms, are possible:
    1. Computing the root of a sum of squares (RMSE) corresponds to the Euclidian norm: it is the notion of distance you are familiar with. It is also called the **ℓ2 norm**.
    2. Computing the sum of absolutes (MAE) corresponds to the **ℓ1 norm**. It is sometimes called the Manhattan norm because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.
- The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare, the RMSE performs very well and is generally preferred. 
- A quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute.
- Working with preprocessed attributes is common in Machine Learning, and it is not necessarily a problem, but you should try to understand how the data was computed.
- When you estimate the generalization error using the test set, your estimate will be too optimistic and you will launch a system that will not perform as well as expected. This is called data snooping bias.
- Stratified sampling: the population is divided into homogeneous subgroups called strata, and the right number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population. For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey data of 1000 people in the US would try to maintain this ratio in the sample: 513 female and 487 male.
- Hence, when creating the test set from the dataset, instead of randomly shuffling the data and keeping a percentage of data aside for test data, you should perform stratified sampling so that the test data preserves the similar distribution of important categories as in the actual dataset.

### Discover and Visualize the Data to Gain Insights
- It is a good idea to look at correlation between features and the target. The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north). Finally, coefficients close to zero mean that there is no linear correlation.
Note: The correlation coefficient only measures linear correlations (“if x goes up, then y generally goes up/down”). It may completely miss out on nonlinear relationships (e.g., “if x is close to zero then y generally goes up”).
- Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function, which plots every numerical attribute against every other numerical attribute.
- One last thing you may want to do before actually preparing the data for Machine Learning algorithms is to try out various attribute combinations. Combine different features and see if you can notice some meaningful pattern or observe a better correlation with the target. 

### Prepare the Data for Machine Learning Algorithms
- Scikit-Learn provides a handy class to take care of missing values: Imputer. First, you need to create an Imputer instance, specifying that you want to replace each attribute’s missing values with the median of that attribute. Then you can fit the imputer instance to the training data using the fit() method. 
- Scikit-Learn Design principles:
	1. Consistency: All objects share a consistent and simple interface:  
		(i) Estimators: Any object that can estimate some parameters based on a dataset is called an estimator (e.g., an imputer is an estimator). The estimation itself is performed by the fit() method, and it takes only a dataset as a parameter. Any other parameter needed to guide the estimation process is considered a hyperparameter (such as an imputer’s strategy), and it must be set as an instance variable.  
		(ii) Transformers: Some estimators (such as an imputer) can also transform a dataset; these are called transformers. Once again, the API is quite simple: the transformation is performed by the transform() method with the dataset to transform as a parameter. It returns the transformed dataset. This transformation generally relies on the learned parameters, as is the case for an imputer. All transformers also have a convenience method called fit_transform().  
		(iii) Predictors: Finally, some estimators are capable of making predictions given a dataset; they are called predictors. A predictor has a predict() method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a score() method that measures the quality of the predictions given a test set.  
	2. Inspection: All the estimator’s hyperparameters are accessible directly via public instance variables (e.g., imputer.strategy), and all the estimator’s learned parameters are also accessible via public instance variables with an underscore suffix (e.g., imputer.statistics_).
	3. Nonproliferation of classes: Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.
	4. Composition: Existing building blocks are reused as much as possible. For example, it is easy to create a Pipeline estimator from an arbitrary sequence of transformers followed by a final estimator.
	5. Sensible defaults: Scikit-Learn provides reasonable default values for most parameters, making it easy to create a baseline working system quickly.
- Scikit-learn provides 'LabelEncoder()' for converting categorical features to numeric ones. One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values (but that might not be the case). To fix this issue, a common solution is to create one binary attribute per category: This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).
- NumPy’s reshape() function allows one dimension to be –1, which means “unspecified”: the value is inferred from the length of the array and the remaining dimensions.
- We can apply both transformations (from text categories to integer categories, then from integer categories to one-hot vectors) in one shot using the 'LabelBinarizer' class.
- Although Scikit-Learn provides many useful transformers, you will need to write your own for tasks such as custom cleanup operations or combining specific attributes. You will want your transformer to work seamlessly with Scikit-Learn functionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inheritance), all you need is to create a class and implement three methods: fit() (returning self), transform(), and fit_transform().
- The more you automate the data preparation steps (by writing custom transformers), the more combinations you can automatically try out, making it much more likely that you will find a great combination (and saving you a lot of time).

- One of the most important transformations you need to apply to your data is feature scaling. Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales. Note that scaling the target values is generally not required.
- There are two common ways to get all attributes to have the same scale:
	1. Min-max scaling: (many people call this normalization) is quite simple. The values are shifted and rescaled so that they end up ranging from 0 to 1. 
	2. Standardization is quite different: first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance. Unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms.
	Note that Min-max scaling is more sensitive to outliers than Standardization. 
- As with all the transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to transform the training set and the test set (and new data).
- There maybe many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the 'Pipeline' class to help with such sequences of transformations. The Pipeline constructor takes a list of name/estimator pairs defining a sequence of steps. All but the last estimator must be transformers (i.e., they must have a fit_transform() method). When you call the pipeline’s fit() method, it calls fit_transform() sequentially on all transformers, passing the output of each call as the parameter to the next call, until it reaches the final estimator, for which it just calls the fit() method.
- Scikit-Learn also provides a 'FeatureUnion' class for combining different transformations (like numerical, categorical) into a single pipeline.
- Building a model on top of many other models is called Ensemble Learning, and it is often a great way to push ML algorithms even further.
- You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make.
- Fine-Tuning Your Model:
    1. Grid Search: One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. Scikit-Learn provides 'GridSearchCV' API for this purpose. All you need to do is tell it which hyperparameters you want it to experiment with, and what values to try out, and it will evaluate all the possible combinations of hyperparameter values, using cross-validation.
    2. Randomized Search: The grid search approach is fine when you are exploring relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead. This class can be used in much the same way as the GridSearchCV class,  but instead of trying out all possible combinations, it evaluates a given number of random  combinations by selecting a random value for each hyperparameter at every iteration. 
    3. Ensemble Methods: Another way to fine-tune your system is to try to combine the models that perform best. The group (or “ensemble”) will often perform better than the best individual
model, especially if the individual models make very different types of errors.

## CHAPTER 3: Classification
### Performance Measures
- Measuring Accuracy Using Cross-Validation: Accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).
- Confusion Matrix: The general idea is to count the number of times instances of class A are classified as class B. Each row in a confusion matrix represents an actual class, while each column represents a predicted class. A perfect classifier would have only true positives and true negatives, so its confusion matrix would have nonzero values only on its main diagonal (top left to bottom right). 
- Consider a Cat vs Dog classifier. Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances (i.e out of the images predicted as 'Cats' how many of them actually are 'Cats'), while recall (also known as sensitivity) is the fraction of the total amount of relevant instances that were actually retrieved  (i.e out of the all the actual 'Cats' images how many are predicted as 'Cats').
- Increasing precision reduces recall, and vice versa. This is called the precision/recall tradeoff. You can have a 'decision threshold' which decides up to what extend you allow the precision or recall to exist in your training. So how can you decide which threshold to use? - We can take help from Scikit-learn's 'decision_function' and 'precision_recall_curve()' function to decide the threshold value. Or you can just plot precision directly against recall and decide on the threshold based on the requirement. 
    <blockquote>If someone says “let’s reach 99% precision,” you should ask, “at what recall?”</blockquote>
- The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate. 
- Since the ROC curve is so similar to the precision/recall (or PR) curve, you may wonder how to decide which one to use. As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives, and the ROC curve otherwise.

**Multilabel Classification**: In some cases you may want your classifier to output multiple classes for each instance. Such a classification system that outputs multiple binary labels is called a multilabel classification system.  

**Multioutput Classification**: It is simply a generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values).


## CHAPTER 4: Training Models
### Linear Regression
- Generally, a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term).
- When using Gradient Descent, you should ensure that all features have a similar scale, or else it will take much longer to converge.
### Gradient Descent
- Training a model means searching for a combination of model parameters that minimizes a cost function (over the training set). It is a search in the model’s parameter space: the more parameters a model has, the more dimensions this space has, and the harder the search is: searching for a needle in a 300-dimensional haystack is much trickier than in three dimensions. Fortunately, since the cost function is convex in the case of Linear Regression, the needle is simply at the bottom of the bowl.
- *Batch Gradient Descent*: it uses the whole batch of training data at every step. As a result it is terribly low on very large training sets. 
- *Stochastic Gradient Descent* just picks a random instance in the training set at every step and computes the gradients based only on that single instance. Obviously this makes the algorithm much faster since it has very little data to manipulate at every iteration. It also makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration. On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much less regular than Batch Gradient Descent: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, never settling down. So once the algorithm stops, the final parameter values are good, but not optimal.
- When the cost function is very irregular, this can actually help the algorithm jump out of local minima, so Stochastic Gradient Descent has a better chance of finding the global minimum than Batch Gradient Descent does. Therefore randomness is good to escape from local optima, but bad because it means that the algorithm can never settle at the minimum. One solution to this dilemma is to gradually reduce the learning rate. The steps start out large (which helps make quick progress and escape local minima), then get smaller and smaller, allowing the algorithm to settle at the global minimum. This process is called simulated annealing.
- *Mini-batch Gradient Descent*: at each step, instead of computing the gradients based on the full training set (as in Batch GD) or based on just one instance (as in Stochastic GD), Mini-batch GD computes the gradients on small random sets of instances called minibatches. 
- Comparison of algorithms for Linear Regression:  

| Algorithm | Large *m* | Large *n* | Hyperparameters | Scaling Required |
| :--------------- | :----------: | -----------: |------------------: |---------------: |
| Normal Equation | Fast | Slow | 0 | No |
| Batch GD | Slow | Fast | 2 | Yes	|
| Stochastic GD | Fast | Fast | >=2 | Yes |
| Mini-batch GD | Fast | Fast | >=2 | Yes |
- When using Gradient Descent, remember that it is important to first normalize the input feature vectors, or else training may be much slower. 

### Polynomial Regression
- What if your data is actually more complex than a simple straight line? Surprisingly, you can actually use a linear model to fit nonlinear data. A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called Polynomial Regression.
- When there are multiple features, Polynomial Regression is capable of finding relationships between features (which is something a plain Linear Regression model cannot do). This is made possible by the fact that *PolynomialFeatures* (which is a scikit-learn function) also adds all combinations of features up to the given degree. 
- If your model is underfitting the training data, adding more training examples will not help. You need to use a more complex model or come up with better features.
- One way to improve an overfitting model is to feed it more training data until the validation error reaches the training error. 

- **The Bias/Variance Tradeoff**: An important theoretical result of statistics and Machine Learning is the fact that a model’s generalization error can be expressed as the sum of three very different
errors:
	1. Bias: This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data. 
	2. Variance: This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance, and thus to overfit the training data.
	3. Irreducible error: This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers). 

### Regularized Linear Models
- A good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. For example, a simple way to regularize a polynomial model is to reduce the number of polynomial degrees.  
#### 1. Ridge Regression
- Ridge Regression is a regularized version of Linear Regression: a regularization term is added to the cost function. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. Note that the regularization term should only be added to the cost function during training. Once the model is trained, you want to evaluate the model’s performance using the unregularized performance measure. 
- **Note**: It is quite common for the cost function used during training to be different from the performance measure used for testing. Apart from regularization, another reason why they might be different is that a good training cost function should have optimization friendly derivatives, while the performance measure used for testing should be as close as possible to the final objective. A good example of this is a classifier trained using a cost function such as the log loss but evaluated using precision/recall.  
#### 2. Lasso Regression
- Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso Regression) is another regularized version of Linear Regression: just like Ridge Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm. 
- An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., set them to zero). In other words, Lasso Regression automatically performs feature selection and outputs a sparse model (i.e., with few nonzero feature weights).  
#### 3. Elastic Net
- Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio.   
<br>
- So when should you use Linear Regression, Ridge, Lasso, or Elastic Net? It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain Linear Regression. Ridge is a good default, but if you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net since they tend to reduce the useless features’ weights down to zero as we have discussed. In general, Elastic Net is preferred over Lasso since Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.
- **Early Stopping**: A very different way to regularize iterative learning algorithms such as Gradient Descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping. 
### Logistic Regression
- Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an instance belongs to a particular class (e.g., what is the probability that this email is spam?). If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (called the positive class, labeled “1”), or else it predicts that it does not (i.e., it belongs to the negative class, labeled “0”). This makes it a binary classifier.
- Just like a Linear Regression model, a Logistic Regression model computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like the Linear Regression model does, it outputs the logistic of this result
- The logistic—also called the logit, is a sigmoid function (i.e., S-shaped) that outputs a number between 0 and 1.
- Just like the other linear models, Logistic Regression models can be regularized using ℓ1 or ℓ2 penalties. 
### Softmax Regression
- The Logistic Regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers. This is called Softmax Regression, or Multinomial Logistic Regression.
- The Softmax Regression classifier predicts only one class at a time (i.e., it is multiclass, not multioutput) so it should be used only with mutually exclusive classes such as different types of plants. You cannot use it to recognize multiple people in one picture. 
- The objective is to have a model that estimates a high probability for the target class (and consequently a low probability for the other classes). Minimizing the cost function, called the *cross entropy*, should lead to this objective because it penalizes the model when it estimates a low probability for a target class. Cross entropy is frequently used to measure how well a set of estimated class probabilities match the target classes.

## CHAPTER 5: Support Vector Machines
- A Support Vector Machine (SVM) is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection. SVMs are particularly well suited for classification of complex but small- or medium-sized datasets.
- You can think of an SVM classifier as fitting the widest possible street between the classes. This is called large margin classification. 
- Adding more training instances “off the street” will not affect the decision boundary at all: it is fully determined (or “supported”) by the instances located on the edge of the street. These instances are called the 'support vectors'.
- If we strictly impose that all instances be off the street and on the right side, this is called hard margin classification. There are two main issues with hard margin classification. First, it only works if the data is linearly separable, and second it is quite sensitive to outliers.
- To avoid these issues it is preferable to use a more flexible model. The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side). This is called soft margin classification. (In Scikit-Learn’s SVM classes, you can control this balance using the C hyperparameter: a smaller C value leads to a wider street but more margin violations).

### Nonlinear SVM Classification
- Although linear SVM classifiers are efficient and work surprisingly well in many cases, many datasets are not even close to being linearly separable. One approach to handling nonlinear datasets is to add more features, such as polynomial features; in some cases this can result in a linearly separable dataset. 
- Adding polynomial features is simple to implement and can work great with all sorts of Machine Learning algorithms (not just SVMs), but at a low polynomial degree it cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow. 
- Fortunately, when using SVMs you can apply an almost miraculous mathematical technique called the **kernel trick** which makes it possible to get the same result as if you added many polynomial features, even with very high degree polynomials, without actually having to add them. 
- **Kernel trick** is a mathematical technique that implicitly maps instances into a very high-dimensional space (called the feature space), enabling nonlinear classification and regression with Support Vector Machines. (Recall that a linear decision boundary in the high-dimensional feature space corresponds to a complex nonlinear decision boundary in the original space.)
- Another technique to tackle nonlinear problems is to add features computed using a similarity function that measures how much each instance resembles a particular landmark. 
- With so many kernels to choose from, how can you decide which one to use? As a rule of thumb, you should always try the linear kernel first, especially if the training set is very large or if it has plenty of features. If the training set is not too large, you should try the Gaussian RBF kernel as well; it works well in most cases. 

### SVM Regression
- The SVM algorithm is quite versatile: not only does it support linear and nonlinear classification, but it also supports linear and nonlinear regression. The trick is to reverse the objective: instead of trying to fit the largest possible street between two classes while limiting margin violations, SVM Regression tries to fit as many instances as possible on the street while limiting margin violations (i.e., instances off the street). 

## CHAPTER 6: Decision Trees
- Like SVMs, Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multioutput tasks. They are very powerful algorithms, capable of fitting complex datasets. 
- One of the many qualities of Decision Trees is that they require very little data preparation. In particular, they don’t require feature scaling or centering at all.
- A node’s samples attribute counts how many training instances it applies to. A node’s value attribute tells you how many training instances of each class this node applies to. Finally, a node’s gini attribute measures its impurity: a node is “pure” (gini=0) if all training instances it applies to belong to the same class.
- Scikit-Learn uses the CART algorithm, which produces only binary trees: nonleaf nodes always have two children (i.e., questions only have yes/no answers).
- Decision Trees are fairly intuitive and their decisions are easy to interpret. Such models are often called *white box* models. In contrast, Random Forests or neural networks are generally considered *black box* models. They make great predictions, and you can easily check the calculations that they performed to make these predictions; nevertheless, it is usually hard to explain in simple terms why the predictions were made.
- Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train Decision Trees (also called “growing” trees). The idea is really quite simple: the algorithm first splits the training set in two subsets using a single feature *k* and a threshold *tk* (e.g., “petal length ≤ 2.45 cm”). How does it choose *k* and *tk*? It searches for the pair *(k, tk)* that produces the purest subsets (weighted by their size). Once it has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets and so on, recursively. It stops recursing once it reaches the maximum depth (defined by the max_depth hyperparameter), or if it cannot find a split that will reduce impurity. A few other hyperparameters control additional stopping conditions.
### Impurity (Gini vs Entropy)
- By default, the Gini impurity measure is used, but you can select the *entropy* impurity measure instead by setting the criterion hyperparameter to "entropy". The concept of entropy originated in thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered. It later spread to a wide variety of domains, including Shannon’s information theory, where it measures the average information content of a message: entropy is zero when all messages are identical. In Machine Learning, it is frequently used as an impurity measure: a set’s entropy is zero when it contains instances of only one class.
### Regularization
- Decision Trees make very few assumptions about the training data (as opposed to linear models, which obviously assume that the data is linear, for example). If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely, and most likely overfitting it. Such a model is often called a nonparametric model, not because it does not have any parameters (it often has a lot) but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a parametric model such as a linear model has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting). 
- To avoid overfitting the training data, you need to restrict the Decision Tree’s freedom during training. This is called regularization. The regularization hyperparameters depend on the algorithm used, but generally you can at least restrict the maximum depth of the Decision Tree. In Scikit-Learn, this is controlled by the max_depth hyperparameter (the default value is None, which means unlimited). Reducing max_depth will regularize the model and thus reduce the risk of overfitting. 
### Regression
- Decision Trees are also capable of performing regression tasks. The Decision Tree will be similar to the classification tree. The main difference is that instead of predicting a class in each node, it predicts a value.
### Instability of Decision Trees (Limitations)
1. Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation.
2. Decision Trees are very sensitive to small variations in the training data.
- Random Forests can limit this instability by averaging predictions over many trees. 

## CHAPTER 7: Ensemble Learning and Random Forests
- Suppose you ask a complex question to thousands of random people, then aggregate their answers. In many cases you will find that this aggregated answer is better than an expert’s answer. This is called the wisdom of the crowd. Similarly, if you aggregate the predictions of a group of predictors (such as classifiers or regressors), you will often get better predictions than with the best individual predictor. A group of predictors is called an *ensemble*; thus, this technique is called *Ensemble Learning*, and an Ensemble Learning algorithm is called an *Ensemble method*.
- For example, you can train a group of Decision Tree classifiers, each on a different random subset of the training set. To make predictions, you just obtain the predictions of all individual trees, then predict the class that gets the most votes. Such an ensemble of Decision Trees is called a Random Forest, and despite its simplicity, this is one of the most powerful Machine Learning algorithms available today.
- Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy. 

### Hard voting and Soft voting
- A very simple way to create a better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes. This majority-vote classifier is called a *hard voting* classifier.
- If all classifiers are able to estimate class probabilities, then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers. This is called *soft voting*. It often achieves higher performance than hard voting because it gives more weight to highly confident votes.

### Bagging and Pasting
- One way to get a diverse set of classifiers is to use very different training algorithms. Another approach is to use the same training algorithm for every predictor, but to train them on different random subsets of the training set. When sampling is performed with replacement, this method is called *bagging* (short for bootstrap aggregating). When sampling is performed without replacement, it is called *pasting*.
- In other words, both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor. 
- Once all predictors are trained, the ensemble can make a prediction for a new instance by simply aggregating the predictions of all predictors. The aggregation function is typically the statistical mode (i.e., the most frequent prediction) for classification, or the average for regression.

### Out-of-Bag Evaluation
- With bagging, some instances may be sampled several times for any given predictor, while others may not be sampled at all. By default a BaggingClassifier samples *m* training instances with replacement, where m is the size of the training set. This means that only about 63% of the training instances are sampled on average for each predictor. The remaining 37% of the training instances that are not sampled are called out-of-bag (oob) instances. 
- Since a predictor never sees the oob instances during training, it can be evaluated on these instances, without the need for a separate validation set or cross-validation. You can evaluate the ensemble itself by averaging out the oob evaluations of each predictor.

### Random Forests
- A Random Forest is an ensemble of Decision Trees, generally trained via the bagging method (or sometimes pasting).
- The Random Forest algorithm introduces extra randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features. This results in a greater tree diversity, which trades a higher bias for a lower variance, generally yielding an overall better model.
- **Feature Importance**: If you look at a single Decision Tree, important features are likely to appear closer to the root of the tree, while unimportant features will often appear closer to the leaves (or not at all). It is therefore possible to get an estimate of a feature’s importance by computing the average depth at which it appears across all trees in the forest.
- Random Forests are very handy to get a quick understanding of what features actually matter, in particular if you need to perform feature selection.

### Boosting
- Boosting refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.
1. **AdaBoost**: One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. This is the technique used by Ada‐Boost.
2. **Gradient Boosting**: Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor. 

### Stacking
- It is based on a simple idea: instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation?
- Example: Consider performing a regression task on a new instance. Each of the predictors predicts a different value, and then the final predictor called a blender, or a meta learner) takes these predictions as inputs and makes the final prediction.
- The trick is to split the training set into three subsets: the first one is used to train the first layer, the second one is used to create the training set used to train the second layer (using predictions made by the predictors of the first layer), and the third one is used to create the training set to train the third layer (using predictions made by the predictors of the second layer). Once this is done, we can make a prediction for a new instance by going through each layer sequentially

## CHAPTER 8: Dimensionality Reduction
- Many Machine Learning problems involve thousands or even millions of features for each training instance. Not only does this make training extremely slow, it can also make it much harder to find a good solution. This problem is often referred to as the *curse of dimensionality*.
- Reducing dimensionality does lose some information (just like compressing an image to JPEG can degrade its quality), so even though it will speed up training, it may also make your system perform slightly worse. It also makes your pipelines a bit more complex and thus harder to maintain. So you should first try to train your system with the original data before considering using dimensionality
reduction if training is too slow. In some cases, however, reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance (but in general it won’t; it will just speed up training).
- The more dimensions the training set has, the greater the risk of overfitting it.
### Main Approaches for Dimensionality Reduction
**1. Projection**: In most real-world problems, training instances are not spread out uniformly across all dimensions. Many features are almost constant, while others are highly correlated. As a result, all training instances actually lie within (or close to) a much lower-dimensional subspace of the high-dimensional space. If we project every training instance perpendicularly from a higher dimensional space onto a lower-dimensional subspace, we can reduce the dimensionality.  
**2. Manifold Learning**: Put simply, a 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space. More generally, a *d*-dimensional manifold is a part of an *n*-dimensional space (where d < n) that locally resembles a d-dimensional hyperplane. Many dimensionality reduction algorithms work by modeling the manifold on which the training instances lie; this is called *Manifold Learning*. It relies on the manifold hypothesis, which holds that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This assumption is very often empirically observed.

### PCA
- Principal Component Analysis (PCA) is by far the most popular dimensionality reduction algorithm. First it identifies the hyperplane that lies closest to the data, and then it projects the data onto it.
- (**Hyperplane**: In geometry, a hyperplane is a subspace whose dimension is one less than that of its ambient space. If a space is 3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional, its hyperplanes are the 1-dimensional lines.)
- The idea of PCA is to choose the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis. 
- PCA identifies the axis that accounts for the largest amount of variance in the training set. It will find out as many axes as the number of dimensions in the dataset. The unit vector that defines the *i*<sup>th</sup> axis is called the *i*<sup>th</sup> principal component (PC). 
- So how can you find the principal components of a training set? Luckily, there is a standard matrix factorization technique called Singular Value Decomposition (SVD) that can decompose the training set matrix X into the dot product of three matrices: **U · Σ · V<sup>T</sup>,** where V<sup>T</sup> contains all the principal components that we are looking for.
- Once you have identified all the principal components, you can reduce the dimensionality of the dataset down to d dimensions by projecting it onto the hyperplane defined by the first d principal components. Selecting this hyperplane ensures that the projection will preserve as much variance as possible.
- Another very useful piece of information is the explained variance ratio of each principal component. It indicates the proportion of the dataset’s variance that lies along the axis of each principal component.
- **Choosing the Right Number of Dimensions**: Instead of arbitrarily choosing the number of dimensions to reduce down to, it is generally preferable to choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%). Unless, of course, you are reducing dimensionality for data visualization—in that case you will generally want to reduce the dimensionality down to 2 or 3.
- As we now know, PCA can be used to compress the original dataset into much lesser dimensions. However, we can also try to decompress the reduced dataset (using inverse transformation of PCA). Of course this won’t give you back the original data, since the projection lost a bit of information, but it will likely be quite close to the original data. The mean squared distance between the original data and the reconstructed data (compressed and then decompressed) is called the *reconstruction error*.  
- **Incremental PCA**: One problem with the preceding implementation of PCA is that it requires the whole training set to fit in memory in order for the SVD algorithm to run. Fortunately, Incremental PCA (IPCA) algorithms have been developed: you can split the training set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is useful for large training sets, and also to apply PCA online (i.e., on the fly, as new instances arrive).  
- **Randomized PCA**: Scikit-Learn offers yet another option to perform PCA, called Randomized PCA. This is a stochastic algorithm that quickly finds an approximation of the first d principal components. It is dramatically faster than the previous algorithms. 
### Kernel PCA
- Kernel trick (that we learnt about in SVM) can also be applied to PCA, making it possible to perform complex nonlinear projections for dimensionality reduction. This is called Kernel PCA (kPCA). It is often good at preserving clusters of instances after projection, or sometimes even unrolling datasets that lie close to a twisted manifold. 
- As kPCA is an unsupervised learning algorithm, there is no obvious performance measure to help you select the best kernel and hyperparameter values. However, dimensionality reduction is often a preparation step for a supervised learning task (e.g., classification), so you can simply use grid search to select the kernel and hyperparameters that lead to the best performance on that task.
- Another approach, this time entirely unsupervised, is to select the kernel and hyperparameters that yield the lowest reconstruction error. However, reconstruction is not as easy as with linear PCA.

### LLE
- Locally Linear Embedding (LLE) is another very powerful nonlinear dimensionality reduction (NLDR) technique. It is a Manifold Learning technique that does not rely on projections like the previous algorithms. 
- In a nutshell, LLE works by first measuring how each training instance linearly relates to its closest neighbors, and then looking for a low-dimensional representation of the training set where these local relationships are best preserved. This makes it particularly good at unrolling twisted manifolds, especially when there is not too much noise. 

### Other Dimensionality Reduction Techniques
- **Multidimensional Scaling** (MDS) reduces dimensionality while trying to preserve the distances between the instances.
- **Isomap** creates a graph by connecting each instance to its nearest neighbors, then reduces dimensionality while trying to preserve the geodesic distances9 between the instances.
- **t-Distributed Stochastic Neighbor Embedding** (t-SNE) reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space (e.g., to visualize the MNIST images in 2D).
- **Linear Discriminant Analysis** (LDA) is actually a classification algorithm, but during training it learns the most discriminative axes between the classes, and these axes can then be used to define a hyperplane onto which to project the data. The benefit is that the projection will keep classes as far apart as possible, so LDA is a good technique to reduce dimensionality before running another classification algorithm such as an SVM classifier.


=============================================================================================================================================
# Hands on ML Notes - Part 2 (Neural Networks and Deep Learning)

## CHAPTER 9: Up and Running with TensorFlow
- TensorFlow is a powerful open source software library for numerical computation, particularly well suited and fine-tuned for large-scale Machine Learning. Its basic principle is simple: you first define in Python a graph of computations to perform, and then TensorFlow takes that graph and runs it efficiently using optimized C++ code.
- Most importantly, it is possible to break up the graph into several chunks and run them in parallel across multiple CPUs or GPUs. TensorFlow also supports distributed computing, so you can train colossal neural networks on humongous training sets in a reasonable amount of time by splitting the computations across hundreds of servers.
- TensorFlow was designed to be flexible, scalable, and production-ready, and existing frameworks arguably hit only two out of the three of these. 
- A TensorFlow program is typically split into two parts: the first part builds a computation graph (this is called the construction phase), and the second part runs it (this is the execution phase). The construction phase typically builds a computation graph representing the ML model and the computations required to train it. The execution phase generally runs a loop that evaluates a training step repeatedly (for example, one step per mini-batch), gradually improving the model parameters. 
- The inputs and outputs are multidimensional arrays, called tensors (hence the name “tensor flow”). Just like NumPy arrays, tensors have a type and a shape. In fact, in the Python API tensors are simply represented by NumPy ndarrays. They typically contain floats, but you can also use them to carry strings (arbitrary byte arrays). 

## CHAPTER 10: Introduction to Artificial Neural Networks
- Artificial neuron: it has one or more binary (on/off) inputs and one binary output. The artificial neuron simply activates its output when more than a certain number of its inputs are active.
### Perceptron
- The Perceptron is one of the simplest ANN architectures. It is based on a slightly different artificial neuron called a **linear threshold unit** (LTU): the inputs and output are now numbers (instead of binary on/off values) and each input connection is associated with a weight. The LTU computes a weighted sum of its inputs (z = w1 x1 + w2 x2 + ⋯ + wn xn = wT · x), then applies a step function to that sum and outputs the result: hw(x) = step (z) = step (wT · x). Training an LTU means finding the right values for w0, w1, and w2.
- A single LTU can be used for simple linear binary classification. It computes a linear combination of the inputs and if the result exceeds a threshold, it outputs the positive class or else outputs the negative class (just like a Logistic Regression classifier or a linear SVM).
- A Perceptron is simply composed of a single layer of LTUs, with each neuron connected to all the inputs. These connections are often represented using special passthrough neurons called input neurons: they just output whatever input they are fed. Moreover, an extra bias feature is generally added (x0 = 1). This bias feature is typically represented using a special type of neuron called a bias neuron, which just outputs 1 all the time.
- *“Cells that fire together, wire together.”* - Hebb's Rule
- The decision boundary of each output neuron is linear, so Perceptrons are incapable of learning complex patterns (just like Logistic Regression classifiers). However, if the training instances are linearly separable, this algorithm would converge to a solution. This is called the Perceptron convergence theorem. 
- Contrary to Logistic Regression classifiers, Perceptrons do not output a class probability; rather, they just make predictions based on a hard threshold. This is one of the good reasons to prefer Logistic Regression over Perceptrons. 
- Some of the limitations of Perceptrons can be eliminated by stacking multiple Perceptrons. The resulting ANN is called a Multi-Layer Perceptron (MLP). 

### Multi-Layer Perceptron and Backpropagation
- An MLP is composed of one (passthrough) input layer, one or more layers of LTUs, called hidden layers, and one final layer of LTUs called the output layer. Every layer except the output layer includes a bias neuron and is fully connected to the next layer. When an ANN has two or more hidden layers, it is called a deep neural network (DNN).
- **Backpropagation**: For each training instance the backpropagation algorithm first makes a prediction (forward pass), measures the error, then goes through each layer in reverse to measure the error contribution from each connection (reverse pass), and finally slightly tweaks the connection weights to reduce the error (Gradient Descent step). 
- In order for this algorithm to work properly, the authors made a key change to the MLP’s architecture: they replaced the step function with the logistic function, `σ(z)=1/(1+exp(–z))`. This was essential because the step function contains only flat segments, so there is no gradient to work with (Gradient Descent cannot move on a flat surface), while the logistic function has a well-defined nonzero derivative everywhere, allowing Gradient Descent to make some progress at every step. The backpropagation algorithm may be used with other activation functions, instead of the logistic function.
- Two other popular activation functions are:
    1. The hyperbolic tangent function `tanh(z)=2σ(2z)–1`. Just like the logistic function it is S-shaped, continuous, and differentiable, but its output value ranges from –1 to 1 (instead of 0 to 1 in the case of the logistic function), which tends to make each layer’s output more or less normalized (i.e., centered around 0) at the beginning of training. This often helps speed up convergence.
    2. The ReLU function `ReLU(z)=max(0, z)`. It is continuous but unfortunately not differentiable at z = 0 (the slope changes abruptly, which can make Gradient Descent bounce around). However, in practice it works very well and has the advantage of being fast to compute. Most importantly, the fact that it does not have a maximum output value also helps reduce some issues during Gradient Descent.
- An MLP is often used for classification, with each output corresponding to a different binary class (e.g., spam/ham, urgent/not-urgent, and so on). When the classes are exclusive (e.g., classes 0 through 9 for digit image classification), the output layer is typically modified by replacing the individual activation functions by a shared softmax function. 
- Biological neurons seem to implement a roughly sigmoid (S-shaped) activation function, so researchers stuck to sigmoid functions for a very long time. But it turns out that the ReLU activation function generally works better in ANNs. This is one of the cases where the biological analogy was misleading.

### Fine-Tuning Neural Network Hyperparameters
- The flexibility of neural networks is also one of their main drawbacks: there are many hyperparameters to tweak. Not only can you use any imaginable network topology (how neurons are interconnected), but even in a simple MLP you can change the number of layers, the number of neurons per layer, the type of activation function to use in each layer, the weight initialization logic, and much more.
#### Number of Hidden Layers: 
- It has actually been shown that an MLP with just one hidden layer can model even the most complex functions provided it has enough neurons. For a long time, these facts convinced researchers that there was no need to investigate any deeper neural networks. But they overlooked the fact that *deep networks have a much higher parameter efficiency than shallow ones*: they can model complex functions using exponentially fewer neurons than shallow nets, making them much faster to train. 
- DNNs take advantage of the fact that lower hidden layers model low-level structures (e.g., line segments of various shapes and orientations), intermediate hidden layers combine these low-level structures to model intermediate-level structures (e.g., squares, circles), and the highest hidden layers and the output layer combine these intermediate structures to model high-level structures (e.g., faces). 
- Not only does this hierarchical architecture help DNNs converge faster to a good solution, it also improves their ability to generalize to new datasets.
- In summary, for many problems you can start with just one or two hidden layers and it will work just fine. For more complex problems, you can gradually ramp up the number of hidden layers, until you start overfitting the training set. 
#### Number of Neurons per Hidden Layer
- The number of neurons in the input and output layers is determined by the type of input and output your task requires.
- As for the hidden layers, a common practice is to size them to form a funnel, with fewer and fewer neurons at each layer—the rationale being that many low-level features can coalesce into far fewer high-level features. However, this practice is not as common now, and you may simply use the same size for all hidden layers. 
- In general you will get more bang for the buck by increasing the number of layers than the number of neurons per layer.
- A simpler approach is to pick a model with more layers and neurons than you actually need, then use early stopping to prevent it from overfitting 
#### Activation Functions
- In most cases you can use the ReLU activation function in the hidden layers. It is a bit faster to compute than other activation functions, and Gradient Descent does not get stuck as much on plateaus, thanks to the fact that it does not saturate for large input values. 
- For the output layer, the softmax activation function is generally a good choice for classification tasks (when the classes are mutually exclusive). For regression tasks, you can simply use no activation function at all.

## CHAPTER 11: Training Deep Neural Nets
### Vanishing/Exploding Gradients Problems
- The backpropagation algorithm works by going from the output layer to the input layer, propagating the error gradient on the way. Once the algorithm has computed the gradient of the cost function with regards to each parameter in the network, it uses these gradients to update each parameter with a Gradient Descent step. 
- Unfortunately, gradients often get smaller and smaller as the algorithm progresses down to the lower layers. As a result, the Gradient Descent update leaves the lower layer connection weights virtually unchanged, and training never converges to a good solution. This is called the ***vanishing gradients*** problem. 
- In some cases, the opposite can happen: the gradients can grow bigger and bigger, so many layers get insanely large weight updates and the algorithm diverges. This is the ***exploding gradients*** problem, which is mostly encountered in recurrent neural networks.
- More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds. (This was one of the reasons why deep neural networks were mostly abandoned for a long time). 
#### Suspects for causing Vanishing/Exploding Gradients Problems
- Few suspects for causing this problems included the combination of the popular logistic sigmoid **activation function** and the **weight initialization technique** that was most popular at the time, namely random initialization using a normal distribution with a mean of 0 and a standard deviation of 1.
- In short, with this activation function and this initialization scheme, the variance of the outputs of each layer is much greater than the variance of its inputs. Going forward in the network, the variance keeps increasing after each layer until the activation function saturates at the top layers. This is actually made worse by the fact that the logistic function has a mean of 0.5, not 0. 
#### Xavier and He initialization
- An initialization strategy called **Xavier initialization** can be used alleviate this issue. The idea is this: We need the signal to flow properly in both directions: in the forward direction when making predictions, and in the reverse direction when backpropagating gradients. We don’t want the signal to die out, nor do we want it to explode and saturate. For the signal to flow properly, the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs,2 and we also need the gradients to have equal variance before and after flowing through a layer in the reverse direction. This initialization strategy for the ReLU activation function (and its variants), is sometimes called **He initialization**.

#### A look at different Activation functions
- Initially most people had assumed that sigmoid activation functions must be an excellent choice. But it turns out that other activation functions behave much better in deep neural networks, in particular the ReLU activation function, mostly because it does not saturate for positive values (and also because it is quite fast to compute). 
- Unfortunately, the ReLU activation function is not perfect. It suffers from a problem known as the *dying ReLUs*: during training, some neurons effectively die, meaning they stop outputting anything other than 0. 
- To solve this problem, you may want to use a variant of the ReLU function, such as the **leaky ReLU**, defined as `LeakyReLUα(z)=max(αz, z)`. The hyperparameter α defines how much the function “leaks”: it is the slope of the function for z < 0, and is typically set to 0.01. This small slope ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up. 
- Other Leaky ReLU variants: 
    1. **Randomized leaky ReLU** (RReLU), where α is picked randomly in a given range during training,
    2. **Parametric leaky ReLU** (PReLU), where α is authorized to be learned during training (instead of being a hyperparameter, it becomes a parameter that can be modified by backpropagation like any other parameter).
- **Exponential linear unit** (ELU): It takes on negative values when z < 0, which allows the unit to have an average output closer to 0. This helps alleviate the vanishing gradients problem. It has a nonzero gradient for z < 0, which avoids the dying units issue. 
#### So which activation function should you use for the hidden layers of your deep neural networks? 
- Although your mileage will vary, in general ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic. If you care a lot about runtime performance, then you may prefer leaky ReLUs over ELUs. If you don’t want to tweak yet another hyperparameter, you may just use the default α values suggested earlier (0.01 for the leaky ReLU, and 1 for ELU). If you have spare time and computing power, you can use cross-validation to evaluate other activation functions, in particular RReLU if your network is overfitting, or PReLU if you have a huge training set. 
#### Batch Normalization
- Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee that they won’t come back during training.
- A technique called **Batch Normalization** (BN) was introduced to address the vanishing/exploding gradients problems, and more generally the problem that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. 
- The technique consists of adding an operation in the model just before the activation function of each layer, simply zero-centering and normalizing the inputs, then scaling and shifting the result using two new parameters per layer (one for scaling, the other for shifting). In other words, this operation lets the model learn the optimal scale and mean of the inputs for each layer.
- In order to zero-center and normalize the inputs, the algorithm needs to estimate the inputs’ mean and standard deviation. It does so by evaluating the mean and standard deviation of the inputs over the current mini-batch (hence the name “Batch Normalization”). 
- Batch Normalization also acts like a regularizer, reducing the need for other regularization techniques (such as dropout).
- Batch Normalization does, however, add some complexity to the model. Moreover, there is a runtime penalty: the neural network makes slower predictions due to the extra computations required at each layer. So if you need predictions to be lightning-fast, you may want to check how well plain ELU + He initialization perform before playing with Batch Normalization. 
#### Gradient Clipping
- A popular technique to lessen the exploding gradients problem is to simply clip the gradients during backpropagation so that they never exceed some threshold (this is mostly useful for recurrent neural networks). 

### Reusing Pretrained Layers
- It is generally not a good idea to train a very large DNN from scratch: instead, you should always try to find an existing neural network that accomplishes a similar task to the one you are trying to tackle, then just reuse the lower layers of this network: this is called ***transfer learning***. It will not only speed up training considerably, but will also require much less training data.
- If the input pictures of your new task don’t have the same size as the ones used in the original task, you will have to add a preprocessing step to resize them to the size expected by the original model. More generally, transfer learning will work only well if the inputs have similar low-level features.
- The more similar the tasks are, the more layers you want to reuse (starting with the lower layers). For very similar tasks, you can try keeping all the hidden layers and just replace the output layer.
- If the model was trained using another framework, you will need to load the weights manually (e.g., using Theano code if it was trained with Theano), then assign them to the appropriate variables. This can be quite tedious.
- It is likely that the lower layers of the first DNN have learned to detect low-level features in pictures that will be useful across both image classification tasks, so you can just reuse these layers as they are. It is generally a good idea to “freeze” their weights when training the new DNN: if the lower-layer weights are fixed, then the higher layer weights will be easier to train (because they won’t have to learn a moving target).
- Since the frozen layers won’t change, it is possible to cache the output of the topmost frozen layer for each training instance. Since training goes through the whole dataset many times, this will give you a huge speed boost as you will only need to go through the frozen layers once per training instance (instead of once per epoch).
- **Model Zoos**: Where can you find a neural network trained for a task similar to the one you want to tackle? The first place to look is obviously in your own catalog of models. Another option is to search in a model zoo. Many people train Machine Learning models for various tasks and kindly release their pretrained models to the public. Ex: Tensorflow Models or Caffe Model Zoo etc.
#### Unsupervised Pretraining
- Suppose you want to tackle a complex task for which you don’t have much labeled training data, but unfortunately you cannot find a model trained on a similar task. In that case, you may still be able to perform *unsupervised pretraining*. That is, if you have plenty of unlabeled training data, you can try to train the layers one by one, starting with the lowest layer and then going up, using an unsupervised feature detector algorithm such as Restricted Boltzmann Machines or autoencoders. 
- Each layer is trained on the output of the previously trained layers (all layers except the one being trained are frozen). Once all layers have been trained this way, you can finetune the network using supervised learning (i.e., with backpropagation).
#### Pretraining on an Auxiliary Task
- One last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task. The first neural network’s lower layers will learn feature detectors that will likely be reusable by the second neural network.

### Faster Optimizers
- Training a very large deep neural network can be painfully slow. So far we have seen four ways to speed up training (and reach a better solution): (i) applying a good initialization strategy for the connection weights, (ii) using a good activation function, (iii) using Batch Normalization, and (iv) reusing parts of a pretrained network. 
- Another huge speed boost comes from using a faster optimizer than the regular Gradient Descent optimizer.
#### Momentum optimization
- Imagine a bowling ball rolling down a gentle slope on a smooth surface: it will start out slowly, but it will quickly pick up momentum until it eventually reaches terminal velocity (if there is some friction or air resistance). This is the very simple idea behind Momentum optimization. In contrast, regular Gradient Descent will simply take small regular steps down the slope, so it will take much more time to reach the bottom. 
- Momentum optimization cares a great deal about what previous gradients were: at each iteration, it adds the local gradient to the momentum vector m (multiplied by the learning rate η), and it updates the weights by simply subtracting this momentum vector. In other words, the gradient is used as an acceleration, not as a speed.
- Due to the momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times before stabilizing at the minimum. This is one of the reasons why it is good to have a bit of friction in the system: it gets rid of these oscillations and thus speeds up convergence.
- The one drawback of Momentum optimization is that it adds yet another hyperparameter to tune.
#### Nesterov Accelerated Gradient
- One small variant to Momentum optimization called Nesterov Accelerated Gradient (or Nesterov Accelerated Gradient) is almost always faster than vanilla Momentum optimization. The idea is to measure the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum. 
#### AdaGrad
- Consider the elongated bowl problem again: Gradient Descent starts by quickly going down the steepest slope, then slowly goes down the bottom of the valley. It would be nice if the algorithm could detect this early on and correct its direction to point a bit more toward the global optimum.
- The AdaGrad algorithm achieves this by scaling down the gradient vector along the steepest dimensions.
- In short, this algorithm decays the learning rate, but it does so faster for steep dimensions than for dimensions with gentler slopes. This is called an adaptive learning rate. It helps point the resulting updates more directly toward the global optimum.
- One additional benefit is that it requires much less tuning of the learning rate hyperparameter η.
### RMSProp
- Although AdaGrad slows down a bit too fast and ends up never converging to the global optimum, the RMSProp algorithm fixes this by accumulating only the gradients from the most recent iterations (as opposed to all the gradients since the beginning of training). It does so by using exponential decay in the first step. 
### Adam Optimization
- Adam, which stands for adaptive moment estimation, combines the ideas of Momentum optimization and RMSProp: just like Momentum optimization it keeps track of an exponentially decaying average of past gradients, and just like RMSProp it keeps track of an exponentially decaying average of past squared gradients
- **The conclusion is that you should almost always use Adam optimization.**

### Learning Rate Scheduling
- Finding a good learning rate can be tricky. If you set it way too high, training may actually diverge. If you set it too low, training will eventually converge to the optimum, but it will take a very long time. If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum, never settling down. 
- You may be able to find a fairly good learning rate by training your network several times during just a few epochs using various learning rates and comparing the learning curves. The ideal learning rate will learn quickly and converge to good solution.
- However, you can do better than a constant learning rate: if you start with a high learning rate and then reduce it once it stops making fast progress, you can reach a good solution faster than with the optimal constant learning rate. There are many different strategies to reduce the learning rate during training. These strategies are called *learning schedules*. 

### Avoiding Overfitting Through Regularization
- Deep neural networks typically have tens of thousands of parameters, sometimes even millions. With so many parameters, the network has an incredible amount of freedom and can fit a huge variety of complex datasets. But this great flexibility also means that it is prone to overfitting the training set.
#### Early Stopping
To avoid overfitting the training set, a great solution is early stopping: just interrupt training when its performance on the validation set starts dropping.
- Although early stopping works very well in practice, you can usually get much higher performance out of your network by combining it with other regularization techniques.
#### ℓ1 and ℓ2 Regularization
- You can use ℓ1 and ℓ2 regularization to constrain a neural network’s connection weights (but typically not its biases). 
#### Dropout
- The most popular regularization technique for deep neural networks is arguably dropout.
- It is a fairly simple algorithm: at every training step, every neuron (including the input neurons but excluding the output neurons) has a probability *p* of being temporarily “dropped out,” meaning it will be entirely ignored during this training step, but it may be active during the next step. The hyperparameter *p* is called the dropout rate, and it is typically set to 50%. After training, neurons don’t get dropped anymore.
- The intuition behind dropout is this: Neurons trained with dropout cannot co-adapt with their neighboring neurons; they have to be as useful as possible on their own. They also cannot rely excessively on just a few input neurons; they must pay attention to each of their input neurons. They end up being less sensitive to slight changes in the inputs. In the end you get a more robust network that generalizes better.
- Another way to understand the power of dropout is to realize that a unique neural network is generated at each training step. Since each neuron can be either present or absent, there is a total of 2N possible networks (where N is the total number of droppable neurons). This is such a huge number that it is virtually impossible for the same neural network to be sampled twice. Once you have run a 10,000 training steps, you have essentially trained 10,000 different neural networks (each with just one training instance). These neural networks are obviously not independent since they share many of their weights, but they are nevertheless all different. The resulting neural network can be seen as an averaging ensemble of all these smaller neural networks.
- Dropout does tend to significantly slow down convergence, but it usually results in a much better model when tuned properly. So, it is generally well worth the extra time and effort.
- *Dropconnect* is a variant of dropout where individual connections are dropped randomly rather than whole neurons. In general dropout performs better.
#### Max-Norm Regularization
- Another regularization technique that is quite popular for neural networks is called max-norm regularization: for each neuron, it constrains the weights *w* of the incoming connections such that ∥w∥<sub>2</sub> ≤ r, where *r* is the max-norm hyperparameter and ∥ · ∥<sub>2</sub> is the ℓ2 norm.
- Maxnorm regularization can also help alleviate the vanishing/exploding gradients problems. 
#### Data Augmentation
- One last regularization technique, data augmentation, consists of generating new training instances from existing ones, artificially boosting the size of the training set. This will reduce overfitting, making this a regularization technique. 
- The trick is to generate realistic training instances; ideally, a human should not be able to tell which instances were generated and which ones were not. Moreover, simply adding white noise will not help; the modifications you apply should be learnable (white noise is not).

### Practical Guidelines
- The configuration in below table will work fine in most cases.
| Parameter | Default value |
| :---------------- | :--------------- |
| Initialization | He initialization |
| Activation function | ELU |
| Normalization | Batch Normalization | 
| Regularization | Dropout |
| Optimizer | Adam |
| Learning rate schedule | None |

- This default configuration may need to be tweaked:
    - If you can’t find a good learning rate (convergence was too slow, so you increased the training rate, and now convergence is fast but the network’s accuracy is suboptimal), then you can try adding a learning schedule such as exponential decay.
    - If your training set is a bit too small, you can implement data augmentation.
    - If you need a sparse model, you can add some ℓ1 regularization to the mix (and optionally zero out the tiny weights after training). If you need an even sparser model, you can try using FTRL instead of Adam optimization, along with ℓ1 regularization.
    - If you need a lightning-fast model at runtime, you may want to drop Batch Normalization, and possibly replace the ELU activation function with the leaky ReLU. Having a sparse model will also help. 

## CHAPTER 12: Distributing TensorFlow Across Devices and Servers
- TensorFlow’s support of distributed computing is one of its main highlights compared to other neural network frameworks. It gives you full control over how to split (or replicate) your computation graph across devices and servers, and it lets you parallelize and synchronize operations in flexible ways so you can choose between all sorts of parallelization approaches.

### Multiple Devices on a Single Machine
- You can often get a major performance boost simply by adding GPU cards to a single machine. In fact, in many cases this will suffice; you won’t need to use multiple machines at all.
- If you don’t own any GPU cards, you can use a hosting service with GPU capability such as Amazon AWS.

### Parallel Execution
- When TensorFlow runs a graph, it starts by finding out the list of nodes that need to be evaluated, and it counts how many dependencies each of them has. TensorFlow then starts evaluating the nodes with zero dependencies (i.e., source nodes). If these nodes are placed on separate devices, they obviously get evaluated in parallel. If they are placed on the same device, they get evaluated in different threads, so they may run in parallel too (in separate GPU threads or CPU cores).

### Multiple Devices Across Multiple Servers
- To run a graph across multiple servers, you first need to define a cluster. A cluster is composed of one or more TensorFlow servers, called tasks, typically spread across several machines. Each task belongs to a job. A job is just a named group of tasks that typically have a common role, such as keeping track of the model parameters (such a job is usually named "ps" for parameter server), or performing computations (such a job is usually named "worker").

## CHAPTER 13: Convolutional Neural Networks

- Why not simply use a regular deep neural network with fully connected layers for image recognition tasks instead of CNN? Unfortunately, although this works fine for small images (e.g., MNIST), it breaks down for larger images because of the huge number of parameters it requires. For example, a 100 × 100 image has 10,000 pixels, and if the first layer has just 1,000 neurons (which already severely restricts the amount of information transmitted to the next layer), this means a total of 10 million connections. And that’s just the first layer. CNNs solve this problem using partially connected layers. 

### Convolutional Layer
- The most important building block of a CNN is the convolutional layer: neurons in the first convolutional layer are not connected to every single pixel in the input image (like they were in traditional neural networks), but only to pixels in their receptive fields. In turn, each neuron in the second convolutional layer is connected only to neurons located within a small rectangle in the first layer. 
- This architecture allows the network to concentrate on low-level features in the first hidden layer, then assemble them into higher-level features in the next hidden layer, and so on. This hierarchical structure is common in real-world images, which is one of the reasons why CNNs work so well for image recognition.
- A neuron located in row `i`, column `j` of a given layer is connected to the outputs of the neurons in the previous layer located in rows `i` to `i+fh–1`, columns `j` to `j+fw–1`, where *fh* and *fw* are the height and width of the receptive field. 
- In order for a layer to have the same height and width as the previous layer, it is common to add zeros around the inputs, as shown in the diagram. This is called *zero padding*.
- It is also possible to connect a large input layer to a much smaller layer by spacing out the receptive fields. The distance between two consecutive receptive fields is called the *stride*.
#### Filters
- A neuron’s weights can be represented as a small image the size of the receptive field called filters (or convolution kernels).
- A layer full of neurons using the same filter gives you a *feature map*, which highlights the areas in an image that are most similar to the filter. During training, a CNN finds the most useful filters for its task, and it learns to combine them into more complex patterns (e.g., a cross is an area in an image where both the vertical filter and the horizontal filter are active).
#### Stacking Multiple Feature Maps
- Each convolutional layer is composed of several feature maps of equal sizes, so it is more accurately represented in 3D. Within one feature map, all neurons share the same parameters (weights and bias term), but different feature maps may have different parameters. A neuron’s receptive field is the same as described earlier, but it extends across all the previous layers’ feature maps. 
- In short, a convolutional layer simultaneously applies multiple filters to its inputs, making it capable of detecting multiple features anywhere in its inputs. 
- The fact that all neurons in a feature map share the same parameters dramatically reduces the number of parameters in the model, but most importantly it means that once the CNN has learned to recognize a pattern in one location, it can recognize it in any other location. In contrast, once a regular DNN has learned to recognize a pattern in one location, it can recognize it only in that particular location.
- Moreover, input images are also composed of multiple sublayers: one per color channel. There are typically three: red, green, and blue (RGB). Grayscale images have just one channel, but some images may have much more—for example, satellite images that capture extra light frequencies (such as infrared).
- Unfortunately, convolutional layers have quite a few hyperparameters: you must choose the number of filters, their height and width, the strides, and the padding type (*same* or *valid*). As always, you can use cross-validation to find the right hyperparameter values, but this is very time-consuming.
- Another problem with CNNs is that the convolutional layers require a huge amount of RAM, especially during training, because the reverse pass of backpropagation requires all the intermediate values computed during the forward pass.

### Pooling Layer
- Their goal is to subsample (i.e., shrink) the input image in order to reduce the computational load, the memory usage, and the number of parameters (thereby limiting the risk of overfitting). Reducing the input image size also makes the neural network tolerate a little bit of image shift (location invariance).
- Just like in convolutional layers, each neuron in a pooling layer is connected to the outputs of a limited number of neurons in the previous layer, located within a small rectangular receptive field. You must define its size, the stride, and the padding type, just like before. However, a pooling neuron has no weights; all it does is aggregate the inputs using an aggregation function such as the max or mean.
- A pooling layer typically works on every input channel independently, so the output depth is the same as the input depth. You may alternatively pool over the depth dimension, in which case the image’s spatial dimensions (height and width) remain unchanged, but the number of channels is reduced. 

### CNN Architectures
- Typical CNN architectures stack a few convolutional layers (each one generally followed by a ReLU layer), then a pooling layer, then another few convolutional layers (+ReLU), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper (i.e., with more feature maps) thanks to the convolutional layers. 
- At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a softmax layer that outputs estimated class probabilities). 
- Lets discuss some popular architectures:
	1. **LeNet-5**: The LeNet-5 architecture is perhaps the most widely known CNN architecture. It was created by Yann LeCun in 1998 and widely used for handwritten digit recognition (MNIST).
	2. **AlexNet**: It is quite similar to LeNet-5, only much larger and deeper, and it was the first to stack convolutional layers directly on top of each other, instead of stacking a pooling layer on top of each convolutional layer.
	3. **GoogLeNet**: The GoogLeNet architecture was much deeper than previous CNNs. This was made possible by sub-networks called *inception modules*, which allow GoogLeNet to use parameters much more efficiently than previous architectures: GoogLeNet actually has 10 times fewer parameters than AlexNet (roughly 6 million instead of 60 million).
	4. **ResNet**: Residual Network (or ResNet), uses an extremely deep CNN composed of 152 layers. The key to being able to train such a deep network is to use *skip connections* (also called shortcut connections): the signal feeding into a layer is also added to the output of a
layer located a bit higher up the stack. When training a neural network, the goal is to make it model a target function h(x). If you add the input x to the output of the network (i.e., you add a skip connection), then the network will be forced to model f(x) = h(x) – x rather than h(x). This is called residual learning. When you initialize a regular neural network, its weights are close to zero, so the network
just outputs values close to zero. If you add a skip connection, the resulting network just outputs a copy of its inputs; in other words, it initially models the identity function. If the target function is fairly close to the identity function (which is often the case), this will speed up training considerably. Moreover, if you add many skip connections, the network can start making progress even if several layers have not started learning yet. Thanks to skip connections, the signal can easily make its way across the whole network. The deep residual network can be seen as a stack of residual units, where each residual unit is a small neural network with a skip connection.


## CHAPTER 14: Recurrent Neural Networks
### Recurrent Neurons
- Up to now we have mostly looked at feedforward neural networks, where the activations flow only in one direction, from the input layer to the output layer (except for a few networks in Appendix E). A recurrent neural network looks very much like a feedforward neural network, except it also has connections pointing backward.
- Let’s consider a simplest possible RNN, composed of just one neuron receiving inputs, producing an output, and sending that output back to itself. At each time step t (also called a frame), this recurrent neuron receives the inputs x<sub>(t)</sub> as well as its own output from the previous time step, y<sub>(t-1)</sub>. We can represent this tiny network against the time axis. This is called unrolling the network through time. 
- Each recurrent neuron has two sets of weights: one for the inputs x<sub>(t)</sub> and the other for the outputs of the previous time step, y<sub>(t-1)</sub>. 
#### Memory Cells
- Since the output of a recurrent neuron at time step t is a function of all the inputs from previous time steps, you could say it has a form of memory. A part of a neural network that preserves some state across time steps is called a memory cell (or simply a cell).
#### Input and Output Sequences
- An RNN can simultaneously take a sequence of inputs and produce a sequence of outputs. For example, this type of network is useful for predicting time series such as stock prices: you feed it the prices over the last N days, and it must output the prices shifted by one day into the future (i.e., from N – 1 days ago to tomorrow).
- Alternatively, you could feed the network a sequence of inputs, and ignore all outputs except for the last one. In other words, this is a sequence-to-vector network. For example, you could feed the network a sequence of words corresponding to a movie review, and the network would output a sentiment score (e.g., from –1 [hate] to +1 [love]).
- Conversely, you could feed the network a single input at the first time step (and zeros for all other time steps), and let it output a sequence. This is a vector-to-sequence network. For example, the input could be an image, and the output could be a caption for that image.
- Lastly, you could have a sequence-to-vector network, called an encoder, followed by a vector-to-sequence network, called a decoder. For example, this can be used for translating a sentence from one language to another.

### Training RNNs
- To train an RNN, the trick is to unroll it through time and then simply use regular backpropagation. This strategy is called backpropagation through time (BPTT).

### Deep RNNs
- It is quite common to stack multiple layers of cells. This gives you a deep RNN.
#### The Difficulty of Training over Many Time Steps
- To train an RNN on long sequences, you will need to run it over many time steps,
making the unrolled RNN a very deep network. Just like any deep neural network it
may suffer from the vanishing/exploding gradients problem and take forever to train.
- The simplest and most common solution to this problem is to unroll the RNN only
over a limited number of time steps during training. This is called truncated backpropagation
through time.
- Besides the long training time, a second problem faced by long-running RNNs is the
fact that the memory of the first inputs gradually fades away. Indeed, due to the transformations
that the data goes through when traversing an RNN, some information is
lost after each time step. After a while, the RNN’s state contains virtually no trace of
the first inputs. This can be a showstopper.
- To solve this problem, various types of cells with long-term
memory have been introduced. They have proved so successful that the basic cells are
not much used anymore.

### Long Short-Term Memory (LSTM) Cell
- If you consider the LSTM cell as a black box, it can be used very much like a basic cell, except it will perform much better; training will converge faster and it will detect long-term dependencies in the data.
- LSTM cell looks exactly like a regular cell, except that its state is split in two vectors: h<sub>(t)</sub> and c<sub>(t)</sub> (“c” stands for “cell”). You can think of h<sub>(t)</sub> as the short-term state and c<sub>(t)</sub> as the long-term state. 
- The key idea is that the network can learn what to store in the long-term state, what to throw away, and what to read from it. As the long-term state c<sub>(t-1)</sub> traverses the network from left to right, it first goes through a forget gate, dropping some memories, and then it adds some new memories via the addition operation (which adds the memories that were selected by an input gate). The result c<sub>(t)</sub> is sent straight out, without any further transformation. So, at each time step, some memories are dropped and some memories are added. Moreover, after the addition operation, the long-term state is copied and passed through the tanh function, and then the result is filtered by the output gate. This produces the short-term state h<sub>(t)</sub> (which is equal to the cell’s output for this time step y<sub>(t)</sub>). 
- Now let’s look at where new memories come from and how the gates work. First, the current input vector x(t) and the previous short-term state h<sub>(t-1)</sub> are fed to four different fully connected layers. They all serve a different purpose:
    - The main layer is the one that outputs g<sub>(t)</sub>. It has the usual role of analyzing the current inputs x<sub>(t)</sub> and the previous (short-term) state h<sub>(t-1)</sub>. In a basic cell, there is nothing else than this layer, and its output goes straight out to y<sub>(t)</sub> and h<sub>(t)</sub>. In contrast, in an LSTM cell this layer’s output does not go straight out, but instead it is partially stored in the long-term state.
    - The three other layers are *gate controllers*. Since they use the logistic activation function, their outputs range from 0 to 1. Their outputs are fed to element-wise multiplication operations, so if they output 0s, they close the gate, and if they output 1s, they open it. Specifically:
        - The forget gate (controlled by f<sub>(t)</sub>) controls which parts of the long-term state should be erased.
        - The input gate (controlled by i<sub>(t)</sub>) controls which parts of g<sub>(t)</sub> should be added to the long-term state (this is why we said it was only “partially stored”).
        - Finally, the output gate (controlled by o<sub>(t)</sub>) controls which parts of the long-term state should be read and output at this time step (both to h<sub>(t)</sub>) and y<sub>(t)</sub>.
- In short, an LSTM cell can learn to recognize an important input (that’s the role of the input gate), store it in the long-term state, learn to preserve it for as long as it is needed (that’s the role of the forget gate), and learn to extract it whenever it is needed. This explains why they have been amazingly successful at capturing long-term patterns in time series, long texts, audio recordings, and more.
#### Peephole Connections
- In a basic LSTM cell, the gate controllers can look only at the input x<sub>(t)</sub> and the previous short-term state h<sub>(t-1)</sub>. It may be a good idea to give them a bit more context by letting them peek at the long-term state as well. 
- A variant of LSTM was proposed with extra connections
called *peephole connections*: the previous long-term state c<sub>(t-1)</sub> is added as an
input to the controllers of the forget gate and the input gate, and the current long-term
state c<sub>(t)</sub> is added as input to the controller of the output gate.
### Gated Recurrent Unit (GRU) cell
- The GRU cell is a simplified version of the LSTM cell, and it seems to perform just as well. The main simplifications are:
    - Both state vectors are merged into a single vector h<sub>(t)</sub>.
    - A single gate controller controls both the forget gate and the input gate. If the gate controller outputs a 1, the input gate is open and the forget gate is closed. If it outputs a 0, the opposite happens. In other words, whenever a memory must be stored, the location where it will be stored is erased first. This is actually a frequent variant to the LSTM cell in and of itself. 
    - There is no output gate; the full state vector is output at every time step. However, there is a new gate controller that controls which part of the previous state will be shown to the main layer.
- LSTM or GRU cells are one of the main reasons behind the success of RNNs in recent
years, in particular for applications in natural language processing (NLP).

### Natural Language Processing
- Most of the state-of-the-art NLP applications, such as machine translation, automatic summarization, parsing, sentiment analysis, and more, are now based (at least in part) on RNNs.
#### Word Embeddings
- Before we start, we need to choose a word representation. One option could be to represent each word using a one-hot vector. Suppose your vocabulary contains 50,000 words, then the nth word would be represented as a 50,000-dimensional vector, full of 0s except for a 1 at the nth position. However, with such a large vocabulary, this sparse representation would not be efficient at all. Ideally, you want similar words to have similar representations, making it easy for the model to generalize what it learns about a word to all similar words.
- The most common solution is to represent each word in the vocabulary using a fairly small and dense vector (e.g., 150 dimensions), called an ***embedding***, and just let the neural network learn a good embedding for each word during training. 
- At the beginning of training, embeddings are simply chosen randomly, but during training, backpropagation automatically moves the embeddings around in a way that helps the neural network perform its task. Typically this means that similar words will gradually cluster close to one another, and even end up organized in a rather meaningful way. For example, embeddings may end up placed along various axes that represent gender, singular/plural, adjective/noun, and so on. 
- Once your model has learned good word embeddings, they can actually be reused fairly efficiently in any NLP application.
- In fact, instead of training your own word embeddings, you may want to download pretrained word embeddings. Just like when reusing pretrained layers, you can choose to freeze the pretrained embeddings or let backpropagation tweak them for your application. The first option will speed up training, but the second may lead to slightly higher performance.
- Embeddings are also useful for representing categorical attributes that can take on a large number of different values, especially when there are complex similarities between values. For example, consider professions, hobbies, dishes, species, brands, and so on. 

## CHAPTER 15: Autoencoders
- Autoencoders are artificial neural networks capable of learning efficient representations of the input data, called codings, without any supervision (i.e., the training set is unlabeled). These codings typically have a much lower dimensionality than the input data, making autoencoders useful for dimensionality reduction. 
- More importantly, autoencoders act as powerful feature detectors, and they can be used for unsupervised pretraining of deep neural networks. 
- Lastly, they are capable of randomly generating new data that looks very similar to the training data; this is called a *generative model*. For example, you could train an autoencoder on pictures of faces, and it would then be able to generate new faces. 
- Surprisingly, autoencoders work by simply learning to copy their inputs to their outputs. This may sound like a trivial task, but constraining the network in various ways can make it rather difficult. For example, you can limit the size of the internal representation, or you can add noise to the inputs and train the network to recover the original inputs. These constraints prevent the autoencoder from trivially copying the inputs directly to the outputs, which forces it to learn efficient ways of representing the data. 
- In short, the codings are byproducts of the autoencoder’s attempt to learn the identity function under some constraints.

### Efficient Data Representations
- Expert chess players were able to memorize the positions of all the pieces in a game by looking at the board for just 5 seconds, a task that most people would find impossible. However, this was only the case when the pieces were placed in realistic positions (from actual games), not when the pieces were placed randomly. Chess experts don’t have a much better memory than you and I, they just see chess patterns more easily thanks to their experience with the game. Noticing patterns helps them store information efficiently.
- Just like the chess players in this memory experiment, an autoencoder looks at the inputs, converts them to an efficient internal representation, and then spits out something that (hopefully) looks very close to the inputs. 
- An autoencoder is always composed of two parts: an encoder (or recognition network) that converts the inputs to an internal representation, followed by a decoder (or generative network) that converts the internal representation to the outputs. The outputs are often called the reconstructions since the autoencoder tries to reconstruct the inputs, and the cost function contains a reconstruction loss that penalizes the model when the reconstructions are different from the inputs.
- Because the internal representation has a lower dimensionality than the input data, the autoencoder is said to be undercomplete. An undercomplete autoencoder cannot trivially copy its inputs to the codings, yet it must find a way to output a copy of its inputs. It is forced to learn the most important features in the input data (and drop the unimportant ones). 
- If the autoencoder uses only linear activations and the cost function is the Mean Squared Error (MSE), then it can be shown that it ends up performing Principal Component Analysis. 

### Stacked Autoencoders
- Just like other neural networks we have discussed, autoencoders can have multiple hidden layers. In this case they are called stacked autoencoders (or deep autoencoders). 
- Adding more layers helps the autoencoder learn more complex codings. However, one must be careful not to make the autoencoder too powerful. Imagine an encoder so powerful that it just learns to map each input to a single arbitrary number (and the decoder learns the reverse mapping). Obviously such an autoencoder will reconstruct the training data perfectly, but it will not have learned any useful data representation in the process (and it is unlikely to generalize well to new instances). 
- The architecture of a stacked autoencoder is typically symmetrical with regards to the central hidden layer (the coding layer). To put it simply, it looks like a sandwich. 
#### Tying Weights
- When an autoencoder is neatly symmetrical, a common technique is to *tie the weights* of the decoder layers to the weights of the encoder layers. This halves the number of weights in the model, speeding up training and limiting the risk of overfitting. 
#### Training One Autoencoder at a Time
- Rather than training the whole stacked autoencoder in one go, it is often much faster to train one shallow autoencoder at a time, then stack all of them into a single stacked autoencoder (hence the name). This is especially useful for very deep autoencoders. 
- During the first phase of training, the first autoencoder learns to reconstruct the inputs. During the second phase, the second autoencoder learns to reconstruct the output of the first autoencoder’s hidden layer. Finally, you just build a big sandwich using all these autoencoders (i.e., you first stack the hidden layers of each autoencoder, then the output layers in reverse order). This gives you the final stacked autoencoder. You could easily train more autoencoders this way, building a very deep stacked autoencoder. 
#### Visualizing the Reconstructions
- One way to ensure that an autoencoder is properly trained is to compare the inputs and the outputs. They must be fairly similar, and the differences should be unimportant details. 
#### Visualizing Features
- Once your autoencoder has learned some features, you may want to take a look at them. There are various techniques for this. 
- Arguably the simplest technique is to consider each neuron in every hidden layer, and find the training instances that activate it the most. This is especially useful for the top hidden layers since they often capture relatively large features that you can easily spot in a group of training instances that contain them. For example, if a neuron strongly activates when it sees a cat in a picture, it will be pretty obvious that the pictures that activate it the most all contain cats. However, for lower layers, this technique does not work so well, as the features are smaller and more abstract, so it’s often hard to understand exactly what the neuron is getting all excited about. 
- Another technique is to feed the autoencoder a random input image, measure the activation of the neuron you are interested in, and then perform backpropagation to tweak the image in such a way that the neuron will activate even more. If you iterate several times (performing gradient ascent), the image will gradually turn into the most exciting image (for the neuron). This is a useful technique to visualize the kinds of inputs that a neuron is looking for. 
- Finally, if you are using an autoencoder to perform unsupervised pretraining—for example, for a classification task—a simple way to verify that the features learned by the autoencoder are useful is to measure the performance of the classifier. 

### Unsupervised Pretraining Using Stacked Autoencoders
- We already know that, if you are tackling a complex supervised task but you do not have a lot of labeled training data, one solution is to find a neural network that performs a similar task, and then reuse its lower layers. This makes it possible to train a high-performance model using only little training data because your neural network won’t have to learn all the low-level features; it will just reuse the feature detectors learned by the existing net. 
- Similarly, if you have a large dataset but most of it is unlabeled, you can first train a stacked autoencoder using all the data, then reuse the lower layers to create a neural network for your actual task, and train it using the labeled data. 

### Denoising Autoencoders
- Another way to force the autoencoder to learn useful features is to add noise to its inputs, training it to recover the original, noise-free inputs. This prevents the autoencoder from trivially copying its inputs to its outputs, so it ends up having to find patterns in the data. 
- The noise can be pure Gaussian noise added to the inputs, or it can be randomly switched off inputs, just like in dropout. 

### Sparse Autoencoders
- Another kind of constraint that often leads to good feature extraction is *sparsity*: by adding an appropriate term to the cost function, the autoencoder is pushed to reduce the number of active neurons in the coding layer. For example, it may be pushed to have on average only 5% significantly active neurons in the coding layer. This forces the autoencoder to represent each input as a combination of a small number of activations. As a result, each neuron in the coding layer typically ends up representing a useful feature (if you could speak only a few words per month, you would probably try to make them worth listening to). 

### Variational Autoencoders
- They are quite different from all the autoencoders we have discussed so far, in particular:
    - They are probabilistic autoencoders, meaning that their outputs are partly determined by chance, even after training (as opposed to denoising autoencoders, which use randomness only during training). 
    - Most importantly, they are generative autoencoders, meaning that they can generate new instances that look like they were sampled from the training set. 
- Both these properties make them rather similar to Restricted Boltzmann Machines, but they are easier to train and the sampling process is much faster. 
- A variational autoencoder has the basic structure of all autoencoders, with an encoder followed by a decoder but there is a twist: instead of directly producing a coding for a given input, the encoder produces a mean coding *μ* and a standard deviation *σ*. The actual coding is then sampled randomly from a Gaussian distribution with mean *μ* and standard deviation *σ*. After that the decoder just decodes the sampled coding normally. 

### Other Autoencoders
- **Contractive autoencoder (CAE)**: The autoencoder is constrained during training so that the derivatives of the codings with regards to the inputs are small. In other words, two similar inputs must have similar codings. 
- **Stacked convolutional autoencoders**: Autoencoders that learn to extract visual features by reconstructing images processed through convolutional layers. 
- **Generative stochastic network (GSN)**: A generalization of denoising autoencoders, with the added capability to generate data. 
- **Winner-take-all (WTA) autoencoder**: During training, after computing the activations of all the neurons in the coding layer, only the top k% activations for each neuron over the training batch are preserved, and the rest are set to zero. Naturally this leads to sparse codings. Moreover, a similar WTA approach can be used to produce sparse convolutional autoencoders. 
- **Adversarial autoencoders**: One network is trained to reproduce its inputs, and at the same time another is trained to find inputs that the first network is unable to properly reconstruct. This pushes the first autoencoder to learn robust codings. 

## CHAPTER 16: Reinforcement Learning
### Learning to Optimize Rewards
- In Reinforcement Learning, a *software agent* makes *observations* and takes *actions* within an *environment*, and in return it receives *rewards*. Its objective is to learn to act in a way that will maximize its expected long-term rewards. 
- You can think of positive rewards as pleasure, and negative rewards as pain (the term “reward” is a bit misleading in this case). In short, the agent acts in the environment and learns by trial and error to maximize its pleasure and minimize its pain. 

### Policy Search
- The algorithm used by the software agent to determine its actions is called its policy. For example, the policy could be a neural network taking observations as inputs and outputting the action to take. 
- The policy can be any algorithm you can think of, and it does not even have to be deterministic. For example, consider a robotic vacuum cleaner whose reward is the amount of dust it picks up in 30 minutes. Its policy could be to move forward with some probability p every second, or randomly rotate left or right with probability 1 – p. The rotation angle would be a random angle between –r and +r. Since this policy involves some randomness, it is called a *stochastic policy*. The robot will have an erratic trajectory, which guarantees that it will eventually get to any place it can reach and pick up all the dust. 
- How would you train such a robot? There are just two policy parameters you can tweak: the probability *p* and the angle range *r*. One possible learning algorithm could be to try out many different values for these parameters, and pick the combination that performs best. This is an example of *policy search*, in this case using a brute force approach. However, when the *policy space* is too large (which is generally the case), finding a good set of parameters this way is like searching for a needle in a gigantic haystack. 
- Another way to explore the policy space is to use genetic algorithms. For example, you could randomly create a first generation of 100 policies and try them out, then “kill” the 80 worst policies and make the 20 survivors produce 4 offspring each. 
- An offspring is just a copy of its parent plus some random variation. The surviving policies plus their offspring together constitute the second generation. You can continue to iterate through generations this way, until you find a good policy. 
- Yet another approach is to use optimization techniques, by evaluating the gradients of the rewards with regards to the policy parameters, then tweaking these parameters by following the gradient toward higher rewards (gradient ascent). This approach is called *policy gradients*(PG). 
For example, going back to the vacuum cleaner robot, you could slightly increase p and evaluate whether this increases the amount of dust picked up by the robot in 30 minutes; if it does, then increase p some more, or else reduce p. 

### Introduction to OpenAI Gym
- One of the challenges of Reinforcement Learning is that in order to train an agent, you first need to have a working environment. If you want to program an agent that will learn to play an Atari game, you will need an Atari game simulator. If you want to program a walking robot, then the environment is the real world and you can directly train your robot in that environment, but this has its limits: if the robot falls off a cliff, you can’t just click “undo.” You can’t speed up time either; adding more computing power won’t make the robot move any faster. And it’s generally too expensive to train 1,000 robots in parallel. In short, training is hard and slow in the real world, so you
generally need a simulated environment at least to bootstrap training. 
- OpenAI gym is a toolkit that provides a wide variety of simulated environments (Atari games, board games, 2D and 3D physical simulations, and so on), so you can train agents, compare them, or develop new RL algorithms. 

### Neural Network Policies
- A neural network policy will take an observation as input, and it will output the action to be executed. More precisely, it will estimate a probability for each action, and then we will select an action randomly according to the estimated probabilities. 
- Why do we pick a random action based on the probability given by the neural network, rather than just picking the action with the highest score. This approach lets the agent find the right balance between exploring new actions and exploiting the actions that are known to work well. Here’s an analogy: suppose you go to a restaurant for the first time, and all the dishes look equally appealing so you randomly pick one. If it turns out to be good, you can increase the probability to order it next time, but you shouldn’t increase that probability up to 100%, or else you will never try out the other dishes, some of which may be even better than the one you tried. 

### Evaluating Actions: The Credit Assignment Problem
- If we knew what the best action was at each step, we could train the neural network as usual, by minimizing the cross entropy between the estimated probability and the target probability. It would just be regular supervised learning. 
- However, in Reinforcement Learning the only guidance the agent gets is through rewards, and rewards are typically sparse and delayed. For example, if the agent manages to accomplish its task in 100 steps, how can it know which of the 100 actions it took were good, and which of them were bad? All it knows is that it chieved the task after the last action, but surely this last action is not entirely responsible. This is called the *credit assignment problem*: when the agent gets a reward, it is hard for it to know which actions should get credited (or blamed) for it. 
- To tackle this problem, a common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a *discount rate* r at each step. 

### Policy Gradients
- As discussed earlier, PG algorithms optimize the parameters of a policy by following the gradients toward higher rewards. One popular class of PG algorithms is called REINFORCE algorithms. Here is one common variant:
    1. First, let the neural network policy play the game several times and at each step compute the gradients that would make the chosen action even more likely, but don’t apply these gradients yet. 
	2. Once you have run several episodes, compute each action’s score (using the method described in the previous paragraph). 
    3. If an action’s score is positive, it means that the action was good and you want to apply the gradients computed earlier to make the action even more likely to be chosen in the future. However, if the score is negative, it means the action was bad and you want to apply the opposite gradients to make this action slightly less likely in the future. The solution is simply to multiply each gradient vector by the corresponding action’s score. 
    4. Finally, compute the mean of all the resulting gradient vectors, and use it to perform a Gradient Descent step. 
- Researchers try to find algorithms that work well even when the agent initially knows nothing about the environment. However, unless you are writing a paper, you should inject as much prior knowledge as possible into the agent, as it will speed up training dramatically. 
- Also, if you already have a reasonably good policy (e.g., hardcoded), you may want to train the neural network to imitate it before using policy gradients to improve it. 

### Markov Decision Processes (MDP)
- In the early 20th century, the mathematician Andrey Markov studied stochastic processes with no memory, called Markov chains. Such a process has a fixed number of states, and it randomly evolves from one state to another at each step. The probability for it to evolve from a state *s* to a state *s′* is fixed, and it depends only on the pair *(s,s′)*, not on past states (the system has no memory). 
- Markov decision processes resemble Markov chains but with a twist: at each step, an agent can choose one of several possible actions, and the transition probabilities depend on the chosen action. Moreover, some state transitions return some reward (positive or negative), and the agent’s goal is to find a policy that will maximize rewards over time. 
- Bellman found a way to estimate the optimal state value of any state *s*, noted `V*(s)`, which is the sum of all discounted future rewards the agent can expect on average after it reaches a state *s*, assuming it acts optimally. He showed that if the agent acts optimally, then the *Bellman Optimality Equation* applies. This recursive equation says that if the agent acts optimally, then the optimal value of the current state is equal to the reward it will get on average after taking one optimal action, plus the expected optimal value of all possible next states that this action can lead to. 
- This algorithm is an example of *Dynamic Programming*, which breaks down a complex problem (in this case estimating a potentially infinite sum of discounted future rewards) into tractable subproblems that can be tackled iteratively (in this case finding the action that maximizes the average reward plus the discounted next state value). 
- Knowing the optimal state values can be useful, in particular to evaluate a policy, but it does not tell the agent explicitly what to do. Luckily, Bellman found a very similar algorithm to estimate the optimal state-action values, generally called *Q-Values*. The optimal Q-Value of the state-action pair *(s,a)*, noted `Q*(s,a)`, is the sum of discounted future rewards the agent can expect on average after it reaches the state *s* and chooses action *a*, but before it sees the outcome of this action, assuming it acts optimally after that action. 

### Temporal Difference Learning and Q-Learning
- Reinforcement Learning problems with discrete actions can often be modeled as Markov decision processes, but the agent initially has no idea what the transition probabilities are, and it does not know what the rewards are going to be either. It must experience each state and each transition at least once to know the rewards, and it must experience them multiple times if it is to have a reasonable estimate of the transition probabilities. 
- The Temporal Difference Learning (TD Learning) algorithm is very similar to the Value Iteration algorithm, but tweaked to take into account the fact that the agent has only partial knowledge of the MDP. In general we assume that the agent initially knows only the possible states and actions, and nothing more. The agent uses an exploration policy—for example, a purely random policy—to explore the MDP, and as it progresses the TD Learning algorithm updates the estimates of the state values based on the transitions and rewards that are actually observed. 

=============================================================================================================================================
# Hands on ML Notes - Appendix
## APPENDIX B: Machine Learning Project Checklist
- This checklist can guide you through your Machine Learning projects. There are eight main steps:
1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to gain insights.
4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.
5. Explore many different models and short-list the best ones.
6. Fine-tune your models and combine them into a great solution.
7. Present your solution.
8. Launch, monitor, and maintain your system.

### 1 Frame the Problem and Look at the Big Picture
1. Define the objective in business terms.
2. How will your solution be used?
3. What are the current solutions/workarounds (if any)?
4. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
5. How should performance be measured?
6. Is the performance measure aligned with the business objective?
7. What would be the minimum performance needed to reach the business objective?
8. What are comparable problems? Can you reuse experience or tools?
9. Is human expertise available?
10. How would you solve the problem manually?
11. List the assumptions you (or others) have made so far.
12. Verify assumptions if possible.

### 2 Get the Data
Note: automate as much as possible so you can easily get fresh data.
1. List the data you need and how much you need.
2. Find and document where you can get that data.
3. Check how much space it will take.
4. Check legal obligations, and get authorization if necessary.
5. Get access authorizations.
6. Create a workspace (with enough storage space).
7. Get the data.
8. Convert the data to a format you can easily manipulate (without changing the data itself).
9. Ensure sensitive information is deleted or protected (e.g., anonymized).
10. Check the size and type of data (time series, sample, geographical, etc.).
11. Sample a test set, put it aside, and never look at it (no data snooping!).

### 3 Explore the Data
**Note:** try to get insights from a field expert for these steps.
1. Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
2. Create a Jupyter notebook to keep a record of your data exploration.
3. Study each attribute and its characteristics:
    - Name
    - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
    - % of missing values
    - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
    - Possibly useful for the task?
    - Type of distribution (Gaussian, uniform, logarithmic, etc.)
4. For supervised learning tasks, identify the target attribute(s).
5. Visualize the data.
6. Study the correlations between attributes.
7. Study how you would solve the problem manually.
8. Identify the promising transformations you may want to apply.
9. Identify extra data that would be useful.
10. Document what you have learned. 

### 4 Prepare the Data
**Notes:**
- Work on copies of the data (keep the original dataset intact).
- Write functions for all data transformations you apply, for five reasons:
    - So you can easily prepare the data the next time you get a fresh dataset
    - So you can apply these transformations in future projects
    - To clean and prepare the test set
    - To clean and prepare new data instances once your solution is live
    - To make it easy to treat your preparation choices as hyperparameters
1. Data cleaning:
    - Fix or remove outliers (optional).
    - Fill in missing values (e.g., with zero, mean, median…) or drop their rows (or columns).
2. Feature selection (optional):
    - Drop the attributes that provide no useful information for the task.
3. Feature engineering, where appropriate:
    - Discretize continuous features.
    - Decompose features (e.g., categorical, date/time, etc.).
    - Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
    - Aggregate features into promising new features.
4. Feature scaling: standardize or normalize features. 

### 5 Short-List Promising Models
**Notes:**
    - If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time (be aware that this penalizes complex models such as large neural nets or Random Forests).
    - Once again, try to automate these steps as much as possible. 
1. Train many quick and dirty models from different categories (e.g., linear, naive Bayes, SVM, Random Forests, neural net, etc.) using standard parameters. 
2. Measure and compare their performance.
    - For each model, use N-fold cross-validation and compute the mean and standard deviation of the performance measure on the N folds.
3. Analyze the most significant variables for each algorithm.
4. Analyze the types of errors the models make.
    - What data would a human have used to avoid these errors?
5. Have a quick round of feature selection and engineering.
6. Have one or two more quick iterations of the five previous steps.
7. Short-list the top three to five most promising models, preferring models that make different types of errors.

### 6 Fine-Tune the System
**Notes:**
    - You will want to use as much data as possible for this step, especially as you move toward the end of fine-tuning. 
    - As always automate what you can.
1. Fine-tune the hyperparameters using cross-validation.
    - Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?).
    - Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach (e.g., using Gaussian process priors).
2. Try Ensemble methods. Combining your best models will often perform better than running them individually.
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.
**Note:** Don’t tweak your model after measuring the generalization error: you would just start overfitting the test set.

### 7 Present Your Solution
1. Document what you have done.
2. Create a nice presentation.
    - Make sure you highlight the big picture first.
3. Explain why your solution achieves the business objective.
4. Don’t forget to present interesting points you noticed along the way.
    - Describe what worked and what did not.
    - List your assumptions and your system’s limitations.
5. Ensure your key findings are communicated through beautiful visualizations or easy-to-remember statements (e.g., “the median income is the number-one predictor of housing prices”).

### 8 Launch!
1. Get your solution ready for production (plug into production data inputs, write unit tests, etc.).
2. Write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops.
    - Beware of slow degradation too: models tend to “rot” as data evolves.
    - Measuring performance may require a human pipeline (e.g., via a crowdsourcing service).
    - Also monitor your inputs’ quality (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale). This is particularly important for online learning systems.
3. Retrain your models on a regular basis on fresh data (automate as much as possible). 

## APPENDIX E: Other Popular ANN Architectures
### Hopfield Networks
- They are associative memory networks: you first teach them some patterns, and then when they see a new pattern they (hopefully) output the closest learned pattern. This has made them useful in particular for character recognition before they were outperformed by other approaches. You first train the network by showing it examples of character images (each binary pixel maps to one neuron), and then when you show it a new character image, after a few iterations it outputs the closest learned character. 
- They are fully connected graphs; that is, every neuron is connected to every other neuron. 
- The training algorithm works by using Hebb’s rule: for each training image, the weight between two neurons is increased if the corresponding pixels are both on or both off, but decreased if one pixel is on and the other is off. 
- To show a new image to the network, you just activate the neurons that correspond to active pixels. The network then computes the output of every neuron, and this gives you a new image. You can then take this new image and repeat the whole process. After a while, the network reaches a stable state. Generally, this corresponds to the training image that most resembles the input image. 
- A so-called energy function is associated with Hopfield nets. At each iteration, the energy decreases, so the network is guaranteed to eventually stabilize to a low-energy state. The training algorithm tweaks the weights in a way that decreases the energy level of the training patterns, so the network is likely to stabilize in one of these lowenergy configurations. Unfortunately, some patterns that were not in the training set also end up with low energy, so the network sometimes stabilizes in a configuration that was not learned. These are called spurious patterns. 
- Another major flaw with Hopfield nets is that they don’t scale very well—their memory capacity is roughly equal to 14% of the number of neurons. 

### Boltzmann Machines
- Just like Hopfield nets, they are fully connected ANNs, but they are based on stochastic neurons: instead of using a deterministic step function to decide what value to output, these neurons output 1 with some probability, and 0 otherwise. The probability function that these ANNs use is based on the Boltzmann distribution (used in statistical mechanics) hence their name. 
- Neurons in Boltzmann machines are separated into two groups: visible units and hidden units. All neurons work in the same stochastic way, but the visible units are the ones that receive the inputs and from which outputs are read. 
- Because of its stochastic nature, a Boltzmann machine will never stabilize into a fixed configuration, but instead it will keep switching between many configurations. If it is left running for a sufficiently long time, the probability of observing a particular configuration will only be a function of the connection weights and bias terms, not of the original configuration. When the network reaches this state where the original configuration is “forgotten,” it is said to be in thermal equilibrium (although its configuration keeps changing all the time). By setting the network parameters appropriately, letting the network reach thermal equilibrium, and then observing its state, we can simulate a wide range of probability distributions. This is called a *generative model*. 
- Training a Boltzmann machine means finding the parameters that will make the network approximate the training set’s probability distribution. 
- Such a generative model can be used in a variety of ways. For example, if it is trained on images, and you provide an incomplete or noisy image to the network, it will automatically “repair” the image in a reasonable way. You can also use a generative model for classification. Just add a few visible neurons to encode the training image’s class (e.g., add 10 visible neurons and turn on only the fifth neuron when the training image represents a 5). Then, when given a new image, the network will automatically turn on the appropriate visible neurons, indicating the image’s class (e.g., it will turn on the fifth visible neuron if the image represents a 5). 
- Unfortunately, there is no efficient technique to train Boltzmann machines. However, fairly efficient algorithms have been developed to train restricted Boltzmann machines (RBM). 

### Restricted Boltzmann Machines
- An RBM is simply a Boltzmann machine in which there are no connections between visible units or between hidden units, only between visible and hidden units. - A very efficient training algorithm, called *Contrastive Divergence* is used in RBM. 
- The great benefit of this algorithm it that it does not require waiting for the network to reach thermal equilibrium: it just goes forward, backward, and forward again, and that’s it. This makes it incomparably more efficient than previous algorithms, and it was a key ingredient to the first success of Deep Learning based on multiple stacked RBMs. 

### Deep Belief Nets
- Several layers of RBMs can be stacked; the hidden units of the first-level RBM serves as the visible units for the second-layer RBM, and so on. Such an RBM stack is called a deep belief net (DBN). 
- Just like RBMs, DBNs learn to reproduce the probability distribution of their inputs, without any supervision. However, they are much better at it, for the same reason that deep neural networks are more powerful than shallow ones: real-world data is often organized in hierarchical patterns, and DBNs take advantage of that. Their lower layers learn low-level features in the input data, while higher layers learn high-level features. 
- Just like RBMs, DBNs are fundamentally unsupervised, but you can also train them in a supervised manner by adding some visible units to represent the labels. Moreover, one great feature of DBNs is that they can be trained in a semisupervised fashion. 
- One great benefit of this semisupervised approach is that you don’t need much labeled training data. If the unsupervised RBMs do a good enough job, then only a small amount of labeled training instances per class will be necessary. Similarly, a baby learns to recognize objects without supervision, so when you point to a chair and say “chair,” the baby can associate the word “chair” with the class of objects it has already learned to recognize on its own. You don’t need to point to every single chair and say “chair”; only a few examples will suffice. 
- The generative capability of DBNs is quite powerful. For example, it has been used to automatically generate captions for images, and vice versa: first a DBN is trained (without supervision) to learn features in images, and another DBN is trained (again without supervision) to learn features in sets of captions (e.g., “car” often comes with “automobile”). Then an RBM is stacked on top of both DBNs and trained with a set of images along with their captions; it learns to associate high-level features in images with high-level features in captions. Next, if you feed the image DBN an image of a car, the signal will propagate through the network, up to the top-level RBM, and back down to the bottom of the caption DBN, producing a caption. Due to the stochastic nature of RBMs and DBNs, the caption will keep changing randomly, but it will generally be appropriate for the image. If you generate a few hundred captions, the most frequently generated ones will likely be a good description of the image. 

### Self-Organizing Maps
- Self-organizing maps (SOM) are quite different from all the other types of neural networks we have discussed so far. They are used to produce a low-dimensional representation of a high-dimensional dataset, generally for visualization, clustering, or classification. 
- The neurons are spread across a map (typically 2D for visualization, but it can be any number of dimensions you want), and each neuron has a weighted connection to every input. 
- Once the network is trained, you can feed it a new instance and this will activate only one neuron: the neuron whose weight vector is closest to the input vector. In general, instances that are nearby in the original input space will activate neurons that are nearby on the map. 
- This makes SOMs useful for visualization (in particular, you can easily identify clusters on the map), but also for applications like speech recognition. For example, if each instance represents the audio recording of a person pronouncing a vowel, then different pronunciations of the vowel “a” will activate neurons in the same area of the map, while instances of the vowel “e” will activate neurons in another area, and intermediate sounds will generally activate intermediate neurons on the map. 
- The training algorithm is unsupervised. It works by having all the neurons compete against each other. First, all the weights are initialized randomly. Then a training instance is picked randomly and fed to the network. All neurons compute the distance between their weight vector and the input vector (this is very different from the artificial neurons we have seen so far). The neuron that measures the smallest distance wins and tweaks its weight vector to be even slightly closer to the input vector, making it more likely to win future competitions for other inputs similar to this one. 
It also recruits its neighboring neurons, and they too update their weight vector to be slightly closer to the input vector (but they don’t update their weights as much as the winner neuron). Then the algorithm picks another training instance and repeats the process, again and again. This algorithm tends to make nearby neurons gradually specialize in similar inputs. 


```python

```
