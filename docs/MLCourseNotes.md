# Machine Learning, Stanford University, Andrew Ng
These are the notes I have prepared while doing the online course for Machine Learning from Coursera taught by Dr. Andrew Ng from Stanford University.

### Table of Contents
* [What is Machine Learning](#define_machine_learning)
    * [Supervised Learning](#define_supervised_learning)
    * [Unsupervised Learning](#define_unsupervised_learning)
* [Linear Regression](#linear_regression_with_one_variable)
    * [Cost Function](#linear_regression_with_one_variable_cost_function)
    * [Gradient Descent](#gradient_descent_algorithm)
    * [Gradient Descent Intuition](#gradient_descent_intuition)
    * [Gradient Descent For Linear Regression](#gradient_descent_linear_regression)
* [Linear regression with multiple variable](#linear_regression_with_multiple_variables)
    * [Multivariate Linear Regression](#multivariate_linear_regression)
    * [Gradient descent in practice 1: Feature Scaling](#gradient_descent_featurescaling)
    * [Gradient descent in practice 2: Learning Rate](#gradient_descent_learningrate)
    * [Features and Polynomial Regression](#features_polynomial_regression)
    * [Normal Equation](#normal_equation)
 * [Logistic Regression](#logistic_regression)
    * [Hypothesis Representation](#hypothesis_representation)
    * [Decision Boundary](#decision_boundary)
    * [Cost Function](#logistic_regression_cost_function)
    * [Simplified Cost Function and Gradient Descent](#simplified_cost_function)
    * [Advanced Optimization](#advanced_optimization)
    * [Multiclass Classification](#multi_class_classification)
 * Regularization
    * [Problem of Overfitting](#problem_of_overfitting)
    * [Cost Function](#cost_function)
    * [Regularized Linear Regression](#regularized_linear_regression)
    * [Regularized Logistic Regression](#regularized_logistic_regression)
 * Neural Networks: Representation
    * [Model Representation](#neural_networks_model_representation)
    * [Multiclass Classification](#multiclass_classification)
 * Neural Networks: Learning
    * [Cost Function](#neural_networks_cost_function)
    * [Backpropagation Algorithm](#backpropagation_algorithm)
    * [Backpropagation Intuition](#backpropagation_intuition)
    * [Implementation Note: Unrolling parameters](#neural_networks_unrolling_params)
    * [Gradient checking](#neural_networks_gradient_checking)
    * [Random Initialization](#neural_networks_theta_initialization)
 * Advice for Applying Machine Learning
    * [Evaluating a Hypothesis](#evaluate_learning_algorithms)
    * [Model Selection and Train/Validation/Test sets](#evaluate_learning_algorithms_model_selection)
    * [Diagnosing Bias vs Variance](#bias_vs_variance)
    * [Regularization and Bias/Variance](#regularization_bias_vs_variance)
    * [Learning Curves](#learning_curves)
 * Machine Learning System Design
    * [Error Analysis](#error_analysis)
    * [Error Metrics for Skewed Classes](#error_metrics_for_skewed_classes)
    * [Trading off precision and recall](#trading_precision_recall)
    * [Data For Machine Learning](#data_for_ml)
 * Support Vector Machines
    * [Optimization Objective](#svm_optimization_objective)
    * [Kernels](#kernels)
    * [Using An SVM](#using_an_svm)
 * Unsupervised Learning
    * [Introduction](#unsupervised_learning)
    * [K-means algorithm](#k_means)
    * [Optimization objective](#k_means_optimization_objective)
    * [Random Initialization](#k_means_random_initialization)
    * [Choosing the number of clusters](#k_means_choosing_number_of_clusters)
 * Dimensionality Reduction
    * [Data Compression](#dimensionality_reduction_data_compression)
    * [Data Visualization](#dimensionality_reduction_data_visualization)
    * [Principal Component Analysis](#pca)
    * Applying PCA 
        * [Reconstruction from compressed Representation](#pca_reconstruction_from_compressed_representation)
        * [Choosing the number of principal components](#pca_choosing_number_of_principal_components)
 * Anomaly Detection
    * [Gaussian distribution](#anomaly_detection_gaussian_distribution)
    * [Algorithm](#anomaly_detection_algorithm)
    * [Developing and evaluating an anomaly detection system](#anomaly_detection_developing)
    * [Anomaly detection vs. Supervised Learning](#anomaly_detection_vs_supervisedlearning)
    * [Choosing What Features to Use](#anomaly_detection_features_to_use)
 * Recommender Systems
    * [Problem Formulation](#recommender_system_problem_formulation) 
    * [Content Based Recommendations](#recommender_system_content_based_recommendation)
    * [Collaborative Filtering](#collaborative_filtering)
    * [Vectorization: Low Rank Matrix Factorization](#low_rank_matrix_factorization)
    * [Implementation Detail: Mean Normalization](#mean_normalization)
* Large Scale Machine Learning
    * [Learning with Large Datasets](#large_datasets)
    * [Stochastic Gradient Descent](#stochastic_gradient_descent)
    * [Mini-Batch Gradient Descent](#minibatch_gradient_descent)
    * [Stochastic Gradient Descent Convergence](#stochastic_gradient_descent_convergence)
    * [Online Learning](#online_learning)
    * [Map Reduce and Data Parallelism](#mapreduce_data_parallelism)
<a name ="define_machine_learning"></a>
## What is Machine Learning

Two definitions of Machine Learning are offered. 
**Arthur Samuel** described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

**Tom Mitchell** provides a more modern definition: "A computer program is said to learn from experience `E` with respect to some class of tasks `T` and performance measure `P`, if its performance at tasks in `T`, as measured by `P`, improves with experience `E`."

Example: playing checkers.

`E` = the experience of playing many games of checkers

`T` = the task of playing checkers.

`P` = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning and Unsupervised learning.

<a name ="define_supervised_learning"></a>
## Supervised Learning
 - Give algorithm data sets where "right answers" are given
 - *Regression problem* - goal is to produce continuous output
 - *Classification problem* - goal is to generate discrete valued output

To summarize:

 In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

 - Supervised learning problems are categorized into `"regression"` and `"classification"` problems. 

 - In a regression problem, we are trying to `predict results within a continuous output`, meaning that we are trying to map input variables to some continuous function. 

 - In a classification problem, we are instead trying to `predict results in a discrete output`. In other words, we are trying to map input variables into discrete categories.

Example 1:
 Given data about the size of houses on the real estate market, try to predict their price. 
 Price as a function of size is a continuous output, so this is a regression problem.
 We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." 
 Here we are classifying the houses based on price into two discrete categories.

Example 2:
 Regression:     Given a picture of a person, we have to predict their age on the basis of the given picture
 Classification: Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.
 
<a name ="define_unsupervised_learning"></a>
## Unsupervised learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. 
We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by *clustering* the data based on relationships among the variables in the data.

With unsupervised learning there is *no feedback* based on the prediction results.

Example:
**Clustering**: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
**Non-clustering**: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

<a name ="linear_regression_with_one_variable"></a>
## Model Representation
### Linear regression with 1 variable
 - In supervised learning we have a data set called as training set, and our job from this data is to learn.
 - Recall that in regression problems, we are taking input variables and trying to fit the output onto a continuous expected result function.
 - Linear regression with one variable is also known as "univariate linear regression."

Univariate linear regression is used when you want to predict a **single output** value y from a **single input** value x. 
We're doing **supervised learning** here, so that means we already have an idea about what the input/output cause and effect should be.

Notation we are going to use for Linear regression:
`m`: no. of training examples
`x`'s : "input" variable / features
`y`'s : "output" variable / "target" variable
`(x, y)`: denotes 1 training example
`(x^(i), y^(i))`: denotes ith training example, please note the `i` is not a exponentiation, that is just an index into the training set, it just refers to the ith row in the training set table.

To describe the supervised learning problem slightly more formally, our goal is, given a `training set`, to learn a function `h : X → Y` so that h(x) is a “good” predictor for the corresponding value of y.
When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem
The use of the `hypothesis` is to take for example size of house(x) -> h -> estimated price(estimated value of y). The function `h` maps from x’s to y’s

### The Hypothesis function
Our hypothesis function has the general form:
`y^=hθ(x)=θ0+θ1x`
We give to `hθ(x)` values for `θ0` and `θ1` to get our estimated output `y^`.
 
Note that we try out various values of `θ0` and `θ1` to try to find values which provide the best possible "fit" or the most representative "straight line" through the data points mapped on the x-y plane.
<a name ="linear_regression_with_one_variable_cost_function"></a>
### Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. This takes an average of all the results of the hypothesis with inputs from x's compared to the actual output y's.
more formally we write it as:
`J(θ0, θ1) = 1/2 * m * [ ∑ i = 1 to m (h(x(i) ) - y(i))^2]` i.e. minimize J(θ0,θ1) over θ0, θ1

simplified version of the previous cost function:
`hθ(x) = θ1(x)`
`J(θ1) = 1/2 * m * [∑ i = 1 to m (θ(x(i)) - y(i))^2]`

minimize J(θ1)
θ0 = 0 means choosing only hypothesis function that passes through origin
*Why do we divide by `m`* ?
Ans: One typical reason to normalize by `m` is so that we can view the cost function as an approximation to the "generalization error", which is the expected square loss on a randomly chosen new example.


- **Hypothesis**: for a fixed `θ1`, is a function of x 
- Cost function `J(θ1)` : is a function of the param `θ1` which controls the slope of the line
Each value of `θ1` corresponds to a different hypothesis or to a different straight line and we could derive a different value of `J(θ1)`

<a name ="gradient_descent_algorithm"></a>
# Gradient Descent algorithm
Given: some function `J(θ0, θ1)`
want: minimize `J(θ0, θ1)` over `(θ0, θ1)`
outline:
- start with some `θ0` ,`θ1`
- keep changing `θ0`, `θ1` to reduce `J(θ0, θ1)` until we hopefully end up at a minimum

Gradient descent is used for solving a more general problem like minimize `J(θ0,..., θn)` over `(θ0,..., θn)`
``` 
repeat until convergence {
θj := θj - α * ∂/∂θj  J(θ0, θ1)  (for j = 0 and j = 1, represent the feature index number)
}
where 
:= used for assignment
= is used for truth assertion
α: learning rate
```
we should be updating θ0 and θ1 simultaneously as follows:

```
temp0 := θ0 - α * ∂/∂θ0  J(θ0, θ1)
temp1 := θ1 - α * ∂/∂θ1  J(θ0, θ1)
θ0 := temp0
θ1 := temp1
```

eg : lets us say θ0 = 1, θ1 = 2 θj := θj + squareroot(θ0 * θ1) for j = 0 and j = 1
```
temp0 := 1 + root(2)
temp1 := 2 + root(2)
θ0 := 1 + root(2)
```
<a name="gradient_descent_intuition"></a>
## Gradient Descent Intuition
### Derivative:
Let us say we have minimize J(θ1) over θ1 and θ1 belongsTo R
`θ1 = θ1 - α * d/dθ1 J(θ1)`

**Computing the derivative:**
The derivative will be a number which determines the slope of the tangent at a given point.
so `θ1 = θ1 - α  * num` if derivative is >= 0

else if derivative is <= 0
`θ1 = θ1 - α  * (- num)`

### Learning rate value:
If alpha(α) is too small, gradient descent can be slow. If alpha is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.
Gradient descent can converge  a local minimum, even with the learning rate fixed
As we approach local minimum, gradient descent will automatically take smaller steps. 
So, no need to decrease alpha over time the derivative will get smaller and smaller and eventually will be 0
<a name="gradient_descent_linear_regression"></a>
## Apply gradient descent to minimize squared error cost function
`∂/∂θj J(θ0, θ1) = ∂/∂θj * 1/2*m ∑i =1 to m [(hθ(x(i)) - y(i))^2]`
i.e.,

`∂/∂θj J(θ0, θ1) = ∂/∂θj * 1/2*m ∑i =1 to m[(θ0 + θ1*x(i) - y(i)) ^2]`
`for j = 0, the expression would be: 1/m * ∑ i = 1 to m[ hθ(x(i)) - y(i) ]`
`for j = 1, the expression would be: 1/m * ∑ i = 1 to m[ hθ(x(i)) - y(i) * x(i) ]`

so, the gradient descent algorithm will be:
```
repeat until convergence {
θ0 := θ0 - α * 1/m  ∑ i =1 to m[ hθ(x(i)) - y(i) ]
θ1 := θ1 - α * 1/m  ∑ i =1 t0 m[ hθ(x(i)) - y(i) * x(i) ]
}

update θ0, θ1 simultaneously.
```

# “Batch”  Gradient descent
“Batch”: each step of gradient descent uses all the training examples.

<a name="multivariate_linear_regression"></a>
# Multivariate Linear Regression:
size | # of beds     |# of floors     |age of home        |price
2104 |4              |5               |23                 |120
1416 |3              |2               |12                 |100

where we use multiple variables such as:
x1 = size
x2 = no.of beds
x3 = no.of floors
x4 = age of home(years)
y = price

hence, we have
n = number of features
`x^(i)` = input features of the `ith` training example
`x^(i)(j)` = value of the feature `j` in `ith` training example
so, in our example x^(2)(2) would be 3, where x^(2) would give us a vector containing the 2nd row as [1416 3 2 12 100]

previously, the hypothesis function was:
`hθ(x) = θ0 + θ1 * x`
 
consider from above the hypothesis function will be:
`hθ(x) = θ0 + θ1 * x1 + θ2 * x2`
for convenience of notation let `x0 = 1` this would mean we have an additional feature or a 0th row

`hθ(x) = θ0 * x0 + θ1 * x1 + ... + θn * xn
           = θT * X `
where.
`θ = [θ0 θ1 θ2 .. θn]` 
`X =[x0 x1 x2 ... xn]`

<a name="linear_regression_with_multiple_variables"></a>
# Linear Regression with Multiple variables
## Gradient descent for multiple variables

A quick summary

*hypothesis:* `hθ(x) = θT * x = θ0 * x0 + θ1 * x1 + θ2 * x2 + . . . + θn * xn`

*parameters:* `θ0, θ1, ..., θn`

*cost function:*
`J(θ0, θ1, θ2, . . ., θn) = 1/2 * m ∑i = 1 to m[ hθ(x(i)) - y(i)^2 ]`

### Gradient descent
**Previously (n = 1)**

Repeat {
θ0 := θ0 - alpha * 1 / m ∑i = 1 to m [hθ(x(i)) - y(i)]

θ1 := θ1 - alpha * 1 / m ∑i = 1 to m [hθ(x(i)) - y(i)] * x(i)
}

**New algorithm (n >= 1)**
Repeat {
θj := θj - α * 1 / m ∑i = 1 to m [hθ(x(i)) - y(i)] * xj^(i) (simultaneous updates for j = 0, . . . , n)
}
<a name="gradient_descent_featurescaling"></a>
## Gradient descent in practice 1: Feature Scaling
- make sure features are on a similar scale, then gradient descent can converge more quickly 
get every feature into approximately a -1 <= x(i) <= 1 range

### Mean normalization
Replace xi with xi - ui to make features have approximately zero mean (do not apply to x0 = 1)
x1 = 72 - 70/22
x2 = 5184 - 6500 / 4000
eg: x1 = size - 1000/2000
x2 = #bedrooms - 2 /5
x1 <- x1- u1/s1 u1: avg value of x1 in training set
s1 is the range i.e. max value - min value or a standard deviation
<a name="gradient_descent_learningrate"></a>
## Gradient descent in practice 2: Learning Rate
### making sure gradient descent is working correctly

- after each iteration your J(θ) value should be decreasing
for example we can make use of automatic convergence test:
declare convergence if J(θ) decreases by less than 10^-3 in 1 iteration
- for sufficiently small value of alpha, J(θ) should decrease on every iteration
- but if alpha is too small, gradient descent can be slow to converge

### Summary:
- if the learning rate is too small: slow convergence
- if learning rate is too large: J(θ) may not decrease on every iteration; may not converge
- to choose learning rate , try: 0.0001, 0.01, 0.1, 1
<a name="features_polynomial_regression"></a>
## Features and polynomial regression

**Features and polynomial regression:**
to fit complicated and non-linear functions
defining new features we may get a new model
we may want to fit in a quadratic model like
θ0 + θ1 * x + θ2 * x ^2 
<a name="normal_equation"></a>
## Computing parameters Analytically
### Normal Equation
method to solve for θ analytically
`θ = (X^T * X)^-1 * X^T * Y`
Octave: `pinv(X' * X) * X' * Y`
the value `θ` minimizes the cost function `J(θ)`
Let us say you have m training examples and n features

|Gradient Descent            | Normal Equation                                       |
|----------------------------|-------------------------------------------------------|
| need to choose alpha       | No need to choose alpha                               |
| need many iterations       | Don't need to iterate                                 |
| works well when n is large | Need to compute (X^T*X)^-1 and slow when n is large   |

# Classification and Representation
## Classification: 
divides the problem in two values i.e. 0(no), 1(yes)
linear regression cannot be applied for classification problems, since we may end up getting values < 1 and < 0, hence we make use of Logistic regression
<a name="logistic_regression"></a> 
## Logistic Regression Hypothesis Representation
the function to represent the hypothesis when we have a classification problem, it has the property that the output or predictions are always between 0 and 1.

**Logistic Regression Model**
We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. 
However, it is easy to construct examples where this method performs very poorly. 
Intuitively, it also doesn’t make sense for hθ(x) to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. 
To fix this, let’s change the form for our hypotheses hθ(x) to satisfy 0≤hθ(x)≤1. 
This is accomplished by plugging θTx into the Logistic Function.
want 0 <= h(θ)(x) <= 1
`hθ(x) = g(θ^T * x)`
where `g(z) = 1/1 + e^-z`

where e = sigmoid/logistic function;sigmoid and logistic are acronyms, and can be used interchangeably
 
`hθ(x) = 1 / 1 + (e ^ -θ^T*x)`
<a name="hypothesis_representation"></a>
**Interpretation of Hypothesis Output**
hθ(x) = estimated probability that y = 1 on input x
For example:
If x = [
        x0  
        x1 
        ]
        
given as: [
           1 
           tumorSize
           ]
we substitute in our hypothesis model to predict `y`, and we get h(x) = 0.7, i.e. probability of `y` being 1 is 0.7, or tell the patient that there is 70% chance of a
tumor being malignant

i.e. `hθ(x) = p(y = 1 | x; θ)` //probability that y = 1, given x, parameterized by θ

since this is a classification problem y can take values either 0 or 1

`p(y = 0 | x; θ) + p(y = 1 | x; θ) = 1`
i.e.
`p(y = 0 | x; θ) = 1 - p(y = 1 | x; θ)` 
<a name="decision_boundary"></a>
## Decision Boundary

we know that,
`hθ(x) = g(θ^T*x)` - (1)
`g(z) = 1 / 1 + e ^-z` - (2)

Suppose, we predict y = 1 if hθ(x) >= 0.5
this would mean g(z) >= 0.5 when z > 0
i.e. hθ(x) = g(θ^T*x) >= 0.5 from (1)
whenever θ^T*x >= 0.5

suppose, we predict y = 0 if if hθ(x) < 0.5
this would mean g(z) < 0.5 when z < 0
i.e. hθ(x) = g(θ^T*x) < 0.5 from (2)
whenever θ^T*x < 0.5

The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.
Decision boundary is a property of the hypothesis.
## Non-Linear decision boundaries
let us suppose hθ(x) = g(θ0 + θ1x1 + θ2x2 + θ3x1^2 + θ4x2^2), and the θ vector is
θ = [ -1
      0
      0
      1
      1
     ]
This means our hypothesis will predict y=1 if -1 + x1^2 + x2^2 >=0
i.e. x1^2 + x2^2 >=1, if we observe the term x1^2 + x2^2 represents a circle, so outside the circle y=1, and inside we have y=0
Once again the training set does not define the decision boundary but the hypothesis does.
     
## Summary:
1. To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

2. The classification problem is just like the regression problem, except that the values y we now want to predict take on only a small number of discrete values. 
For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) 
For instance, if we are trying to build a spam classifier for email, then x(i) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. 
Hence, y∈{0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” 
Given x(i), the corresponding y(i) is also called the label for the training example.
3. Remember, 
```
z=0,e0=1⇒g(z)=1/2
z→∞,e−∞→0⇒g(z)=1
z→−∞,e∞→∞⇒g(z)=0
``` 
so if our input to g is θ^TX, then
`hθ(x)=g(θTx)≥0.5 when θTx≥0` 
hence, we can say
`θTx≥0⇒y=1`
`θTx<0⇒y=0`

4. Again, the input to the sigmoid function g(z) (e.g. θTX) doesn't need to be linear, and could be a function that describes a circle (e.g. z=θ0+θ1x21+θ2x22) or any shape to fit our data.
<a name="logistic_regression_cost_function"></a> 
# Logistic Regression
## Cost Function
Given:
Training set: {(x(1), y(1)), (x(2), y(2)),..., (x(m), y(m))}
m examples `x` ∈ [x0
                x1
                ... 
                xn]
where x0 = 1 where `x` is also called as feature vector, and y ∈ {0, 1} , and
hθ(x) = 1 / 1 + e^-θT x

**How to choose parameters θ ?**
We know for Liner regression, the cost function looks like:
`J(θ) = 1/m ∑ i = 1 to m[ 1/2 (hθ(x(i)) - y(i))^2]`
alternatively, we can say
`J(θ) = 1/m ∑ i = 1 to m[ cost(hθ(x(i) - y(i))]` 

where cost(hθ(x(i) - y(i)) = 1/2 (hθ(x(i)) - y(i))^2
to simplify more, 
`cost(hθ(x) - y) = 1/2 (hθ(x) - y)^2`
it is the cost to be paid if it outputs hθ(x), and its actual value is y
this cost we found is in terms of linear regression, but we are interested in logistic regression
if we use this cost function, it will turn out to be a non-convex function, since the function hθ(x) is non-linear
i.e. if we were to plot the cost function, it would not guarantee to converge to a local minima

```
cost(hθ(x) - y) =  -log(hθ(x)) if y = 1
                   -log(1 - hθ(x)) if y = 0
```

cost = 0, if y = 1, hθ(x) = 1 i.e. we correctly predicted the output y
but as hθ(x) approaches 0, cost approaches infinity
captures intuition that if hθ(x) = 0, that is as good as saying 
P(y = 1 | x;θ) = 0, but if y = 1; then we will penalize learning algorithm by a very large cost

similarly in the case where y = 0, as hθ(x) approaches 1, the cost function blows up and goes to infinity

### Summary of cost function
We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. 
In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:
```
J(θ)=1/m ∑i=1m Cost(hθ(x(i)),y(i))
Cost(hθ(x),y)=−log(hθ(x)) if y = 1
Cost(hθ(x),y)=−log(1−hθ(x)) if y = 0
```

Remember
```
Cost(hθ(x),y)=0 if hθ(x)=y
Cost(hθ(x),y)→∞ if y=0 andhθ(x)→1 
Cost(hθ(x),y)→∞ if y=1 and hθ(x)→0
```

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. 
If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. 
If our hypothesis approaches 0, then the cost function will approach infinity.
 
Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression 
<a name="simplified_cost_function"></a>
## Simplified cost function and gradient descent
We know that our logistic regression cost function looks like:

`J(θ)=1/m ∑i=1m Cost(hθ(x(i)),y(i))`

`Cost(hθ(x),y)=−log(hθ(x)) if y = 1`
`Cost(hθ(x),y)=−log(1−hθ(x)) if y = 0`
Note: y = 0 or 1 always

`cost(hθ(x),y) = -ylog(hθ(x))-(1-y)log(1−hθ(x))`
therefore our cost function now becomes,
```
J(θ)= 1/m ∑i=1m Cost(hθ(x(i)),y(i))
    = - 1/m [∑i=1m y(i)log(hθ(x(i)))+(1-y(i))log(1−hθ(x(i)))]
```
    
Given this cost function, inorder to fit the parameters,
we try to find parameters θ inorder to minimize J(θ)
 if we are given a new value `x`  we output of the hypothesis will be
 `h(θ)(x) = 1/1+e^-θTx` i.e. estimating that probability that y = 1
 
### Gradient Descent
We know that, 
`J(θ)= - 1/m [∑i=1m y(i)log(hθ(x(i)))+(1-y(i))log(1−hθ(x(i)))]`

Want minθ J(θ):
```
Repeat {
 θj:=θj−α ∂/∂θjJ(θ) ; simultaneously update all θj
}
```
after solving the derivative
```
Repeat {

 θj:=θj−α ∑i=1m (hθ(x(i)) - y(i)) x(i)(j) ; simultaneously update all θj
}
```

even though the algorithm looks identical to linear regression, the hypothesis is different
in case of Linear regression our hypothesis was
`hθ(x) = θTx`
but in case of Logistic regression
`h(θ)(x) = 1/1+e^-θTx`

so, if we were running gradient descent to fit a logistic regression model with parameter θ ∈ Rn+1 , a reasonable way to make sure the learning rate `alpha denoted as α`  is set properly and that gradient descent is running correctly,
plot `J(θ)= - 1/m [ ∑i = 1 to m y(i)log(hθ(x(i))) + (1 - y(i))log(1 − hθ(x(i)))]` as a function of the number of iterations and make sure J(θ) is decreasing every iteration

A vectorized implementation of the form `θ:=θ − α∂` for some vector `∂ ∈ Rn+1` it would be
  
`θ:=θ − α ∑i = 1 to m [(hθ(x(i)) - y(i)) x(i)]`

`θ:=θ − α/m X^T(g(Xθ)−y⃗ )`
### Summary
We can compress our cost function's two conditional cases into one case:

`Cost(hθ(x), y) = −ylog(hθ(x)) − (1 − y)log(1 − hθ(x))`
Notice that when y is equal to 1, then the second term `(1 − y)log(1 − hθ(x))` will be zero and will not affect the result. 
If y is equal to 0, then the first term `−ylog(hθ(x))` will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

`J(θ)= −1/m ∑i = 1 to m [y(i)log(hθ(x(i))) + (1 − y(i))log(1 − hθ(x(i)))]`

A vectorized implementation is:

`h=g(Xθ)J(θ)=1/m⋅(−yTlog(h) − (1 − y)Tlog( 1− h))`

General form of gradient descent is:
Repeat {
    θj:= θj − α∂/∂θj J(θ)
}
<a name="advanced_optimization"></a>
## Advanced Optimization

Optimization algorithm:
Cost function J(θ), we want to minθ J(θ):
Given θ we have code that can compute 
- `J(θ)`
- `α∂/∂θj J(θ)` (for j = 0, 1, ..., n)

Optimization algorithms:
- gradient descent
- conjugate gradient
- BFGS
- L-BFGS

inorder to compute we need to have a function that returns a cost function and gradient
θ = [ θ0
      θ1
      ... 
      θn]
          
function [jval, gradient] = costFunction(θ);
jval = [code to compute J(θ)];

gradient(1) = [code to compute ∂/∂θ0 J(θ)];
gradient(2) = [code to compute ∂/∂θ1 J(θ)];
.
.
.
gradient(n + 1) = [code to compute ∂/∂θn J(θ)];

so, suppose you want to use an advanced optimization algorithm to minimize the cost function for logistic regression with parameters θ0 and θ1, we have
```
function [jval, gradient] = costFunction(θ)
jVal = [code to compute J(θ)]
gradient(1) = CODE_1 % derivative for θ_0
gradient(2) = CODE_2 % derivative for θ_1
```
so CODE_1 will be 1/m ∑i = 1 to m [(hθ(x(i)) - y(i)) * x0(i)]
and CODE_2 will be 1/m ∑i = 1 to m [(hθ(x(i)) - y(i)) * x1(i)]

To summarize:
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value θ:
```
J(θ)
∂/∂θj J(θ)
```
we can write a single function that can return both of these:
```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```
Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".
```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```   
We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.
<a name="multi_class_classification"></a>
## Multi-class classification: one-vs-all
Now we will approach the classification of data when we have more than two categories. Instead of y = {0, 1} we will expand our definition so that y = {0, 1...n}.

Since y = {0, 1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.
```
y ∈ {0,1...n}
h(0)θ(x)=P(y=0|x;θ)
h(1)θ(x)=P(y=1|x;θ)
⋯
h(n)θ(x)=P(y=n|x;θ)

prediction=maxi(h(i)θ(x))
```

We are basically choosing one class and then lumping all the others into a single second class. 
We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

Train a logistic regression classifier hθ(x) for each class￼to predict the probability that `y = i`.
To make a prediction on a new `x`, pick the class that **maximizes** `hθ(x)`
<a name="regularization"></a>
# Regularization
<a name="problem_of_overfitting"></a>
## Problem of overfitting
If we have too many features, the learned hypothesis may fit the training set very well (J(theta) = 1/2m ∑i=1m (hθ(x(i)) - y(i))^2 ~ 0), but fail to
generalize to new examples
Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. 
It is usually caused by a function that is too simple or uses too few features. 
At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. 
It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

### Addressing overfitting:
1. reduce number of features
    - manually select features to keep
    - model selection algorithm
2. Regularization
    - keep all the features, but reduce magnitude/values of θj
    - works well when we have a lot of features, each of which contributes a bit to predicting y
<a name="cost_function"></a>
## Cost Function
Small values for parameters θ0, θ1, ..., θn
- simpler hypothesis
- less prone to overfitting

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.
Say we wanted to make the following function more quadratic:

`θ0+θ1x+θ2x^2+θ3x^3+θ4x^4`
We'll want to eliminate the influence of `θ3x^3` and `θ4x^4`. 
Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:

`minθ 1/2m ∑mi=1 (hθ(x(i))−y(i))^2 + 1000⋅θ3^2 + 1000⋅θ4^2`
We've added two extra terms at the end to inflate the cost of θ3 and θ4. 
Now, in order for the cost function to get close to zero, we will have to reduce the values of θ3 and θ4 to near zero. 
This will in turn greatly reduce the values of θ3x^3 and θ4x^4 in our hypothesis function.
We could also regularize all of our theta parameters in a single summation as:

`minθ 1/2m [ ∑i=1 to m (hθ(x(i))−y(i))^2 + λ ∑j=1 to n θj^2 ]`
The `λ`, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated.
Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. 
If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if λ=0 or is too small ?
<a name="regularized_linear_regression"></a>
## Regularized Linear Regression
We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

Gradient Descent

We will modify our gradient descent function to separate out θ0 from the rest of the parameters because we do not want to penalize θ0.
```
Repeat {
θ0:=θ0 − α 1/m ∑i=1 to m (hθ(x(i))−y(i))x(i)0
θj:=θj − α [(1/m ∑i=1 to m (hθ(x(i)) − y(i))x(i)j) + λ/m θj] j ∈ {1,2...n}
}
```
The term `λ/m θj` performs our regularization. With some manipulation our update rule can also be represented as:

`θj := θj(1 − α λ/m)− α 1/m ∑i=1 to m (hθ(x(i)) − y(i))x(i)j`
The first term in the above equation, `1 − αλ/m` will always be less than 1. 
Intuitively you can see it as reducing the value of θj by some amount on every update. 
Notice that the second term is now exactly the same as it was before.

## Normal Equation

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:
```
θ=(XT X + λ⋅L)^ −1 XT y where  L=⎡0
                                    1
                                      1⋱
                                         1⎦

```

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. 
It should have dimension `(n+1)×(n+1)`. 
Intuitively, this is the identity matrix (though we are not including x0), multiplied with a single real number `λ`.
Recall that if `m ≤ n`, then `XTX` is non-invertible. However, when we add the term `λ⋅L`, then `XTX + λ⋅L` becomes invertible.
<a name="regularized_logistic_regression"></a>
## Regularized Logistic Regression 
When using regularized logistic regression the best way to monitor whether gradient descent is working correctly is to 
Plot `- [ 1/m  ∑i=1 to m y(i) log hθ(x(i)) + (1 - y(i)) log(1 - hθ(x(i)))] + λ/2m  ∑j=1 to n θj^2` as a function of number of iterations and make sure it's decreasing

We can regularize logistic regression in a similar way that we regularize linear regression. 
As a result, we can avoid overfitting.

## Cost Function
Recall that our cost function for logistic regression was:

`J(θ) = − 1/m ∑i=1 to m [y(i) log(hθ(x(i)))+(1−y(i)) log(1−hθ(x(i)))]`
We can regularize this equation by adding a term to the end:
    
`J(θ) = − 1/m ∑i=1 to m [y(i) log(hθ(x(i)))+(1−y(i)) log(1−hθ(x(i)))]+ λ/2m ∑j=1 to n θj^2`

The second sum, `∑j=1 to n θj^2` means to explicitly exclude the bias term, θ0. 
I.e. the θ vector is indexed from 0 to n (holding n+1 values, θ0 through θn), and this sum explicitly skips θ0, by running from 1 to n, skipping 0.
Thus, when computing the equation, we should continuously update the two following equations:

```
Repeat {
θ0 := θ0 − α 1/m ∑i=1 to m (hθ(x(i))−y(i))x0(i)
θj:=θj − α [(1/m ∑i=1 to m (hθ(x(i)) − y(i))x(i)j) + λ/m θj] j ∈ {1,2...n}

hθ(x) = 1 / 1 + e^-θTx
}
```

# Neural Networks: Representation
<a name="neural_networks_model_representation"></a>
### Model Representation
Let's examine how we will represent a hypothesis function using neural networks. 
At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). 
In our model, our dendrites are like the input features x1⋯xn, and the output is the result of our hypothesis function. 
In this model our `x0` input node is sometimes called the "bias unit." It is always equal to 1. 
In neural networks, we use the same logistic function as in classification, `1/1+e−θTx`, yet we sometimes call it a sigmoid (logistic) activation function. 
In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

[ x0
  x1   -> [] -> hθ(x)
  x2]

Our input nodes `(layer 1)`, also known as the `"input layer"`, go into another node `(layer 2)`, which finally outputs the hypothesis function, known as the `"output layer"`.
We can have intermediate layers of nodes between the input and output layers called the "hidden layers."
In this example, we label these intermediate or "hidden" layer nodes `a02⋯an2` and call them `"activation units."`
```
ai(j)="activation" of unit i in layer j
Θ(j)=matrix of weights controlling function mapping from layer j to layer j+1
```

If we had one hidden layer, it would look like:
[x0       [a1(2)
            
 x1    ->  a2(2)    -> hθ(x)
 
 x2        a3(2)]
  
 x3]       

The values for each of the "activation" nodes is obtained as follows:
```
a1(2)=g(Θ10(1)x0+Θ11(1)x1+Θ12(1)x2+Θ13(1)x3)
a2(2)=g(Θ20(1)x0+Θ21(1)x1+Θ22(1)x2+Θ23(1)x3)
a3(2)=g(Θ30(1)x0+Θ31(1)x1+Θ32(1)x2+Θ33(1)x3)

hΘ(x)=a1(3)=g(Θ10(2)a0(2)+Θ11(2)a1(2)+Θ12(2)a2(2)+Θ13(2)a3(2))
```

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters.
We apply each row of the parameters to our inputs to obtain the value for one activation node. 
Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix `Θ(2)` containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, `Θ(j)`.

The dimensions of these matrices of weights is determined as follows:

If network has `sj` units in layer `j` and `sj+1` units in layer `j+1`, then `Θ(j)` will be of dimension `sj+1×(sj+1)`.
We're going to define a new variable `z(j)k` that encompasses the parameters inside our `g` function. 
In our previous example if we replaced by the variable `z` for all the parameters we would get:
a1(2)=g(z1(2))
a2(2)=g(z2(2))
a3(2)=g(z3(2))

In other words, for layer j=2 and node k, the variable z will be:
`zk(2)= Θk,0(1)x0 + Θk,1(1)x1 + ⋯ + Θk,n(1)xn`

The vector representation of x and zj is:
x = [x0
     x1
     ...
     xn]
z(j) = [z1(j)
        z2(j)
        ...
        zn(j)]
        
Setting x=a(1), we can rewrite the equation as:
`z(j)=Θ(j−1)a(j−1)`

We are multiplying our matrix `Θ(j−1)` with dimensions `sj×(n+1)` (where `sj` is the number of our activation nodes) by our vector a^(j−1) with height (n+1). 
This gives us our vector `z(j)` with height sj. Now we can get a vector of our activation nodes for layer j as follows:

`a(j)=g(z(j))`
Where our function `g` can be applied element-wise to our vector `z(j)`.
We can then add a bias unit (equal to 1) to layer j after we have computed a(j). 
This will be element a0(j) and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:

`z(j+1)=Θ(j)a(j)`
We get this final z vector by multiplying the next theta matrix after Θ(j−1) with the values of all the activation nodes we just got. 
This last theta matrix Θ(j) will have only one row which is multiplied by one column a(j) so that our result is a single number. 
We then get our final result with:

`hΘ(x)=a(j+1)=g(z(j+1))`
Notice that in this last step, between layer j and layer j+1, we are doing exactly the same thing as we did in logistic regression. 
Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

A simple example of applying neural networks is by predicting x1 AND x2, which is the logical 'and' operator and is only true if both x1 and x2 are 1.

The graph of our functions will look like:
[ x0
  x1   -> [g(z(2))] -> hΘ(x)
  x2]

Remember that x0 is our bias variable and is always 1.

Let's set our first theta matrix as:

Θ(1)=[−30 20 20]
This will cause the output of our hypothesis to only be positive if both x1 and x2 are 1. In other words:
```
hΘ(x)=g(−30+20x1+20x2)
x1=0  and  x2=0  then  g(−30)≈0
x1=0  and  x2=1  then  g(−10)≈0
x1=1  and  x2=0  then  g(−10)≈0
x1=1  and  x2=1  then  g(10)≈1
```
So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. 
Neural networks can also be used to simulate all the other logical gates
The Θ(1) matrices for AND, NOR, and OR are:

AND:
Θ(1)=[−30 20 20]

NOR:
Θ(1)=[10 −20 −20]

OR:
Θ(1)=[−10 20 20]

We can combine these to get the XNOR logical operator (which gives 1 if x1 and x2 are both 0 or both 1).
[x0       [a1(2) 
 x1   ->   a2(2)]   -> [a(3)] -> hΘ(x)
 x2]

For the transition between the first and second layer, we'll use a Θ(1) matrix that combines the values for AND and NOR:
Θ(1)=[−30 20 20
       10 -20 −20]

For the transition between the second and third layer, we'll use a Θ(2) matrix that uses the value for OR:
Θ(2)=[−10 20 20]

hence, the activation nodes values will be:
a(2) = g(Θ(1).x)
a(3) = g(Θ(2).x)
hΘ(x) = a(3)
```
x1 x2 a1(2) a2(2) hΘ(x)
0  0   0     1     1 
0  1   0     0     0
1  0   0     0     0
1  1   1     0     1
```
<a name="multiclass_classification"></a>
# Multiclass Classification
To classify data into multiple classes, we let our hypothesis function return a vector of values. 
Say we wanted to classify our data into one of four categories(pedestrian, car, truck, and motorcycle). 
We will use the following example to see how this classification is done. 
This algorithm takes as input an image and classifies it accordingly:
We can define our set of resulting classes as y:

y(i) = [1   [0  [0  [0
        0    1   0   0
        0    0   1   0
        0]   0]  0]  1]
Each y(i) represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. 
The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:
[x0      [a0(2)      [a0(3)           [hΘ(x)1
 x1       a1(2)       a1(3)            hΘ(x)2
 x2   ->  a2(2)  ->   a2(3) -> ... ->  hΘ(x)3
 ...      ...]        ...]             hΘ(x)4]
 xn]
<a name="neural_networks_cost_function"></a>          
# Cost Function
Let's first define a few variables that we will need to use:

 - L = total number of layers in the network
 - sl = number of units (not counting bias unit) in layer l
 - K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote hΘ(x)k as being a hypothesis that results in the kth output. 
Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. 
Recall that the cost function for regularized logistic regression was:

`J(θ)= −1/m ∑i=1 to m [y(i) log(hθ(x(i))) + (1−y(i)) log(1−hθ(x(i)))] + λ/2m ∑j=1 to n θj^2`

For neural networks, it is going to be slightly more complicated:

`J(Θ)= −1/m ∑i=1 to m ∑k=1 to K [y(i)k log((hΘ(x(i)))k) + (1−y(i)k)log(1−(hΘ(x(i)))k)] + λ/2m∑l=1 to L−1 ∑i=1 to sl ∑j=1 to sl+1(Θ(l)j,i)^2`
We have added a few nested summations to account for our multiple output nodes. 
In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. 
The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). 
The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). 
As before with logistic regression, we square every term

Note:

the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
the triple sum simply adds up the squares of all the individual Θs in the entire network.
the i in the triple sum does not refer to training example i
<a name="backpropagation_algorithm"></a>
## Backpropagation Algorithm
"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

`minΘJ(Θ)`
That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

`∂/∂Θ(l)i,jJ(Θ)`
To do so, we use the following algorithm:

Given training set `{(x(1),y(1))⋯(x(m),y(m))}`
Set `Δ(l)i,j := 0` for all (l,i,j), (hence you end up having a matrix full of zeros)
For training example t=1 to m:

1. Set a(1):=x(t)
2. Perform forward propagation to compute a(l) for l=2,3,…,L
3. Using y(t), compute `δ(L)=a(L)−y(t)`
   Where `L` is our total number of layers and `a(L)` is the vector of outputs of the activation units for the last layer. 
   So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. 
   To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:
   `g′(z(l))=a(l) .∗ (1−a(l))`
4. Compute δ(L−1),δ(L−2),…,δ(2) using `δ(l)=((Θ(l))Tδ(l+1)) .∗ a(l) .∗ (1−a(l))`
    The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. 
    We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by z(l).

The g-prime derivative terms can also be written out as:
`g′(z(l))=a(l) .∗ (1−a(l))`

5. `Δ(l)i,j := Δ(l)i,j + a(l)jδ(l+1)i` or with vectorization, `Δ(l):=Δ(l) + δ(l+1)(a(l))T`
Hence we update our new Δ matrix.

    - `D(l)i,j:=1/m (Δ(l)i,j+λΘ(l)i,j)`, if `j≠0`.
    - `D(l)i,j:=1mΔ(l)i,j` If `j=0`

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. 
Thus we get ∂/∂Θ(l)ijJ(Θ)= D(l)ij
<a name="backpropagation_intuition"></a>
## Intuition:

`J(Θ)=−1/m ∑t=1 to m ∑k=1 to K [y(t)k log(hΘ(x(t)))k+(1−y(t)k) log(1−hΘ(x(t))k)]+λ/2m ∑l=1 to L−1 ∑i=1 to sl ∑j=1 to sl+1(Θ(l)j,i)2`
If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:
`cost(t)=y(t) log(hΘ(x(t))) + (1−y(t)) log(1−hΘ(x(t)))`
Intuitively, δ(l)j is the "error" for a(l)j (unit j in layer l). More formally, the delta values are actually the derivative of the cost function:
`δ(l)j=∂/∂zj(l)cost(t)`

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are.
<a name="neural_networks_unrolling_params"></a>
## Implementation Note: Unrolling parameters
With neural networks, we are working with sets of matrices:

Θ(1),Θ(2),Θ(3),…
D(1),D(2),D(3),…

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
To summarize:
    Have initial params Θ(1),Θ(2),Θ(3)
    Unroll to get `initialTheta` to pass to `fminunc(@costFunction, initialTheta, options)`
    
    function [jval, gradientVec] = costFunction(thetaVec)
    From thetaVec, get Θ(1),Θ(2),Θ(3)
    Use forward prop/back prop to compute D(1),D(2),D(3) and J(Θ)
    Unroll D(1),D(2),D(3) to get gradientVec
<a name="neural_networks_gradient_checking"></a>    
## Gradient checking
Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

∂/∂ΘJ(Θ)≈J(Θ+ϵ)−J(Θ−ϵ)/2ϵ //2ϵ = (J + Θ) - (J - Θ)

With multiple theta matrices, we can approximate the derivative with respect to Θj as follows:

∂∂ΘjJ(Θ)≈J(Θ1,…,Θj+ϵ,…,Θn)−J(Θ1,…,Θj−ϵ,…,Θn)/2ϵ

A small value for ϵ (epsilon) such as ϵ=10^−4, guarantees that the math works out properly. If the value for ϵ is too small, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the Θj matrix. In octave we can do it as follows:
```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```
We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.

Once you have verified that your backpropagation algorithm is correct, you don't need to compute gradApprox again. 
The code to compute gradApprox can be very slow.
<a name="neural_networks_theta_initialization"></a>
## Random Initialization
Initializing all theta weights to zero does not work with neural networks. 
When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our Θ matrices using the following method:

- initialize each Θij(l) to a random value in [ -ϵ, ϵ] i.e. -ϵ <= Θi,j(l) <= ϵ
-  `theta1 = rand(10, 11)*(2*INIT_EPSILON) - INIT_EPSILON;`
   `theta2 = rand(1, 11)*(2*INIT_EPSILON) - INIT_EPSILO;`

Hence, we initialize each Θ(l)ij to a random value between[−ϵ,ϵ]. 
Using the above formula guarantees that we get the desired bound. The same procedure applies to all the Θ's. 
Below is some working code you could use to experiment.
```
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```
`rand(x,y)` is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.

### Summary of Neural Networks
We start by picking a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

Number of input units = dimension of features x(i)
Number of output units = number of classes
Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.
### How to train a Neural Network?
1. Randomly initialize the weights
2. Implement forward propagation to get hΘ(x(i)) for any x(i)
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example:
```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

*NOTE:* Ideally, we want hΘ(x(i)) ≈ y(i). This will minimize our cost function. 
However, keep in mind that J(Θ) is not convex and thus we can end up in a local minimum instead.
<a name="evaluate_learning_algorithms"></a>
# Evaluating a learning algorithm
### Evaluating a Hypothesis
We perform some troubleshooting by:

- Getting more training examples
- Trying smaller sets of features
- Trying additional features
- Trying polynomial features
- Increasing or decreasing λ

A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). 
Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a `training` set and a `test` set. 
Typically, the training set consists of 70 % of your data and the test set is the remaining 30 %.

The new procedure using these two sets is then:
1. Learn Θ and minimize Jtrain(Θ) using the training set
2. Compute the test set error Jtest(Θ)
<a name="evaluate_learning_algorithms_test_set_error"></a>
### The test set error
1. for Linear regression, `Jtest(Θ)=1/2mtest ∑i=1 to mtest(hΘtest(x(i)) − ytest(i))^2`
2. For classification ~ Misclassification error (aka 0/1 misclassification error):
```
    err(hΘ(x),y)=1 if hΘ(x)≥0.5 and y=0 or hΘ(x)<0.5 and y=1
                =0 otherwise
 ```

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:
`Test Error=1/mtest ∑i=1 to mtest err(hΘtest(x(i)),ytest(i))`

This gives us the proportion of the test data that was misclassified.
<a name="evaluate_learning_algorithms_model_selection"></a>
### Model Selection and Train/Validation/Test sets
Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. 
It could over fit and as a result your predictions on the test set would be poor. 
The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. 
In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to break down our dataset into the three sets is:

Training set: 60%
Cross validation set: 20%
Test set: 20%
We can now calculate three separate error values for the three different sets using the following method:

- Optimize the parameters in Θ using the training set for each polynomial degree.
- Find the polynomial degree d with the least error using the cross validation set.
- Estimate the generalization error using the test set with Jtest(Θ(d)), (d = theta from polynomial with lower error);
This way, the degree of the polynomial d has not been trained using the test set.
<a name="bias_vs_variance"></a>
### Diagnosing Bias vs Variance
The degree of the polynomial d might be contributing to underfitting or overfitting the learning algorithm

- High bias is underfitting and high variance is overfitting. Ideally, we need to find a mean between these two.

The training error will tend to **decrease** as we increase the degree d of the polynomial.
The cross validation error will tend to **decrease** as we increase d up to a point, and then it will **increase** as d is increased, forming a convex curve.

- **High bias (underfitting)**: both `Jtrain(Θ)` and `JCV(Θ)` will be high. Also, `JCV(Θ)≈Jtrain(Θ)`.

- **High variance (overfitting)**: `Jtrain(Θ)` will be low and `JCV(Θ)` will be much greater than `Jtrain(Θ)`.
<a name="regularization_bias_vs_variance"></a>
### Regularization and Bias/Variance
As λ increases, our fit becomes more rigid. 
On the other hand, as `λ` approaches 0, we tend to over overfit the data. 
So how do we choose our parameter `λ` to get it 'just right' ? In order to choose the model and the regularization term `λ`, we need to:

 - Create a list of lambdas (i.e. `λ` ∈ {0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
 - Create a set of models with different degrees or any other variants.
 - Iterate through the `λ`s and for each `λ` go through all the models to learn some `Θ`.
 - Compute the cross validation error using the learned `Θ` (computed with `λ`) on the `JCV(Θ)` **without** regularization or `λ = 0`.
 - Select the best combo that produces the lowest error on the cross validation set.
 - Using the best combo `Θ` and `λ`, apply it on `Jtest(Θ)` to see if it has a good generalization of the problem.
<a name="learning_curves"></a>
### Learning Curves
Let us say we have Jtrain, and JCV
`Jtrain(Θ) = 1/2m ∑i=1 to m (hΘ(x(i) - y(i))^2`
`JCV(Θ) = 1/2mcv ∑i=1 to mcv (hΘ(x(i) - y(i))^2`

Training an algorithm on a very few number of data points (such as 1, 2 or 3) will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. 
As the training set gets larger, the error for a quadratic function increases.
The error value will plateau out after a certain m, or training set size.
**Experiencing high bias**:

**Low training set size**: causes `Jtrain(Θ)` to be low and `JCV(Θ)` to be high.

**Large training set size**: causes both `Jtrain(Θ)` and `JCV(Θ)` to be high with `Jtrain(Θ)≈JCV(Θ)`.

If a learning algorithm is suffering from **high bias**, getting more training data will not (by itself) help much
**Experiencing high variance**:

**Low training set size**: `Jtrain(Θ)` will be low and `JCV(Θ)` will be high.

**Large training set size**: `Jtrain(Θ)` increases with training set size and `JCV(Θ)` continues to decrease without leveling off. 
Also, `Jtrain(Θ) < JCV(Θ)` but the difference between them remains significant.

If a learning algorithm is suffering from **high variance**, getting more training data is likely to help.
### Revisit
- Getting more training examples -> fixes high variance
- Trying smaller sets of features -> fixes high variance
- Trying additional features -> fixes high bias
- Trying polynomial features(x1^2, x2^2, x1x2) -> fixes high bias
- Increasing `λ` -> fixes high bias
- Decreasing `λ` -> fixes high variance

**Diagnosing Neural Networks**

- A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
- A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase `λ`) to address the overfitting.
- Using a single hidden layer is a good starting default. 
- You can train your neural network on a number of hidden layers using your cross validation set. 
- You can then select the one that performs best.

**Model Complexity Effects**:

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. 
These have low bias on the training data, but very high variance.
- In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

## Building a spam classifier
- For some learning applications, it is possible to imagine coming up with many different features. But it can be hard to guess in advance which features will be useful
- There are often many possible ideas for how to develop a high accuracy learning system; "gut feeling" is not a recommended one
- Collect lots of data (for example "honeypot" project but doesn't always work)
- Develop sophisticated features (for example: using email header data in spam emails)
- Develop algorithms to process your input in different ways (recognizing misspellings in spam).
It is difficult to tell which of the options will be most helpful.
<a name="error_analysis"></a>
### Error Analysis
- A recommended approach to perform error analysis using cross validation data rather than test data is to develop new features by examining test set, we end up 
choosing features that will work well specifically for the test set, so `Jtest(Θ)` is no longer a good estimate of how well we generalize example
- Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
- Plot learning curves to decide if more data, more features, etc. are likely to help.
- Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
- It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance. 
For example if we use `stemming`, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), 
and get a 3% error rate instead of 5%, then we should definitely add it to our model. 
However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature. 
Hence, we should try new things, get a numerical value for our error rate, and based on our result decide whether we want to keep the new feature or not.
<a name="error_metrics_for_skewed_classes">
## Handling skewed Data

|Precision/Recall  | Actual Class 1 | Actual class 0 |
|------------------|----------------|----------------|
|Predicted Class 1 | True positive  | False positive |
|Predicted Class 0 | False negative | True negative  |

**Precision**: of all patients where we predicted y=1, what fraction actually has cancer?
`True positives / #Predicted positives` = `True positives / True positive + False positive`

**Recall**: of all patients that actually have cancer, hat fraction did we correctly detect as having cancer?
`True positives / #Actual positives` = `True positives / True positive + False negative`
<a name="trading_precision_recall"></a>
### Trading off precision and recall:
we know that, 
precision = true positives/predicted positives
recall = true positives/no.of actual positives

Logistic regression `0 <= hΘ(x) <= 1`
Predict 1 if hΘ(x) >= (say) 0.5
Predict 0 if hΘ(x) < (say) 0.5

suppose we want to predict y=1, only if very confident, then
*high precision, low recall*
suppose we want to avoid missing too many cases of false negatives, then
*high recall, low precision*

more generally, predict 1 if `hΘ(x) >= threshold`

so, if we increase the threshold from 0.5 to 0.7, then
more y=0 predictions, this will increase the decrease of true positives and increase the number of false negatives, so recall will decrease
### F1 score
how to compare precision/recall numbers?
F score = 2 * (precision * recall / precision + recall)
<a name="data_for_ml"></a>
### Large data rationale
Assume feature x belongs to Rn+1 has sufficient information to predict y accurately

 * Use a learning algorithm with many parameters 
    eg: linear regression with many features; neural network with many features
    we have a low bias algorithm, and Jtrain(Θ) will be small
 * Use a very large training set(unlikely to overfit)
   we have a low variance -> Jtrain(Θ) ~~ Jtest(Θ) -> Jtest will be small

## Large Margin Classification
<a name="svm_optimization_objective"></a>
### Support Vector Machines: Optimization Objective
Alternative view of Logistic Regression

Support Vector Machines (SVM) provides a cleaner way to learn non-linear functions
We know that, Hypothesis for Logistic regression is given by `hΘ(x) = 1/1 + e^-ΘTx`
Let z = ΘTx. Now, consider y=1 (either in training set, cross validation set or test set), we want hΘ(x) ~ 1, ΘTx >> 0
conversely, y=0 (either in training set, cross validation set or test set), we want hΘ(x) ~ 0, ΘTx << 0

Cost function for a single training example = `-(yloghΘ(x) + (1-y)log(1 - hΘ(x)))`
after substituting the value for the hypothesis function
= `-ylog 1/1+e^-ΘTx - (1-y)log(1-1/1+e^-ΘTx)`

when y=1, we only have the first term
`cost1(ΘTx(i))= -log1/1+e^-ΘTx` -- (1)

the case when y=0, we have
`cost0(ΘTx(i)) = -log(1-1/1+e^-ΘTx)` -- (2)

Cost function J(Θ), for logistic regression with regularization is given as
`J(Θ) = minΘ 1/m [ ∑i=1 to m y(i)(-loghΘ(x(i))) + (1-y(i))(-log(1-hΘx(i))) ] + λ/2m ∑j=1 to m Θj^2`

replacing the costs from above equations 1 and 2, we get equation for SVM:
`minΘ 1/m [ ∑i=1 to m y(i)cost1(ΘTx(i)) + (1-y(i))cost0(ΘTx(i)) ] + λ/2m ∑j=1 to m Θj^2`
for SVM we write the above equation slightly differently, 
starting with, getting rid of the term `1/m`
<a name="kernels"></a>
## Kernels
Given x, compute new features depending on proximity landmarks l(1), l(2), l(3)
Given an example x, we define the first feature as f1 = similarity(x, l(1)) //some measure of similarity, which is given as
f1 = similarity(x, l(1)) = exp (- ||x-l(1)||^2 / 2 σ^2)
f2 = similarity(x, l(2)) = exp (- ||x-l(2)||^2 / 2 σ^2)
f3 = similarity(x, l(3)) = exp (- ||x-l(3)||^2 / 2 σ^2)
the similarity function here is nothing but the `kernel` function, the one we are using is Gaussian Kernel

we can write the numerator as follows, for example for f1: 
`exp (- ∑j=1 to n (xj - lj(1))^2 / 2 σ^2)`
if x ~~ l(1) ie. x is closer to l(1), then
f1 ~~ exp(- 0^2 / 2 σ^2) ~~ 1

else if x is far from l(1), then
f1 ~~ exp(- large_number^2 / 2 σ^2) ~~ 0

**How to choose landmarks?**
Given `m` training examples
(x(1), y(1)), (x(2), y(2)),..., (x(m), y(m))
we choose the locations of landmarks to be exactly in the same locations of x(1), x(2),..., x(m) i,e. the training examples
so, l(1)=x(1), l(2)=x(2),.., l(m)=x(m)

Given an example x:
f1 = similarity(x, l(1))
f2 = similarity(x, l(m))

we then can represent a feature vector as f = [f0
                                               f1
                                              ... 
                                               fn
                                               ] 
                                                
where f0 = 1
For training example (x(i), y(i))
f1(i) = similarity(x(i), l(1))
f2(i) = similarity(x(i), l(2))
...
fm(i) = similarity(x(i), l(m))
hence, from above there will be one feature where x(i) = l(i) ie. the gaussian kernel exp(-0/2σ^2) = 1
we can have a feature vector to represent your training example as follows:
f = [f0(i)
     f1(i)
    ... 
     fm(i)
    ]
where f0(i) = 1
## SVM with kernels
hypothesis: Given x, compute features f ∈ Rm+1, predict "y=1" if ΘT f >= 0
training:
`minΘ C[ ∑i=1 to m y(i)cost1(ΘTf(i)) + (1-y(i))cost0(ΘTf(i)) ] + 1/2 ∑j=1 to m Θj^2`

## SVM parameters
C(= 1/λ) Large C: lower bias, higher variance, more prone to overfitting (small λ)
         Small C: higher bias, lower variance, more prone to underfitting (large λ)

σ^2 large σ^2: features fi may vary smoothly, higher bias, lower variance
    small σ^2: features fi may vary less smoothly, lower bias, higher variance
<a name="using_an_svm"></a>
## Using an SVM
- use SVM software package to solve for parameters Θ
- Need to specify
    - choice of parameter C
    - choice of kernel(similarity function): either linear kernel or gaussian kernel
In case of multi-class classification, many SVM packages already have built-in multi-class classification functionality
otherwise, use one-vs-all method. Train K SVMs, one to distinguish y=i from the rest, for i=1,2,..., K, get Θ(1), Θ(2), ..., Θ(k)
Pick class i with largest (Θ(i))T x
### Logistic regression vs. SVMs
n = number of features (x ∈ Rn+1)
m = number of training examples
if n is large(relative to m):
 use logistic regression, or SVM without a kernel("linear kernel")

if n is small, m is intermediate:
 use SVM with Gaussian kernel

if n is small, m is large:
 create/add more features, then use logistic regression or SVM without a kernel
 
Neural networks likely to work well for most of these settings, but may be slower to train
<a name="unsupervised_learning"></a> 
# Clustering
## Unsupervised Learning: Introduction
- In unsupervised learning the training set is of the form {x(1), x(2), x(3), ..., x(m)} without labels y(i)
- Clustering is an example of unsupervised learning
- In unsupervised learning, you are given an unlabeled dataset and are asked to find "structure" in the data
<a name="k_means"></a>
## K-means algorithm
- Randomly initialize 2 points named as cluster centroids, we have 2 of them because we want to group data into 2 clusters
- Step 1:
    * Cluster assignment step:
        - go through each data points, and determine which data point is closer to the above centroid and assign them to cluster centroid
    * Move cluster centroids to their means/ move centroid step
        - calculate the average of points or the mean and move the cluster centroid there
- Step 2: repeat step1 till all the points are present in some cluster

More formally
Input:

- K (no.of clusters)
- training set {x(1), x(2), x(3), ..., x(m)} x(i) ∈ Rn (drop x0=1 convention)
- randomly initialize K cluster centroids u1, u2, u3,..., uk ∈ Rn
- repeat {
    //cluster assignment step
    for i=1 to m
     c(i) := index (from 1 to K) of cluster centroid closest to x(i), min k|| x(i) - uk || ^2
     //move centroid
    for k = 1 to K
     uk = average (mean) of points assigned to cluster k
  }
<a name="k_means_optimization_objective"></a>    
## Optimization objective
once again,

* c(i): index of cluster (1, 2, ..., K) to which example x(i) is currently assigned
* uk: cluster centroid k (uk ∈ Rn)
* uc(i): cluster centroid of cluster to which example x(i) has been assigned
eg: let us say x(i) = 5, c(i) = 5 i.e. uc(i) = u5

Optimization objective:
`J(c(1), ..., c(m), u1,..., uk) = 1/m ∑i=1 to m || x(i) - uc(i) || ^2`

K-means algorithm is carrying out optimization in the ints steps, for example in the first step i.e. the cluster assignment step, k-means tries to minimize the cost function
J(...) w.r.t c(1), ..., c(m) keeping u1,..., uk fixed
and in the second step minimizes J(...) w.r.t u1,..., uk
We can use this to debug our algorithm to make sure our k-means is working correctly
<a name="k_means_random_initialization"></a>
## Random Initialization
When running k-means following are the tasks that can eb done:
1. the number of cluster centroids K should be less than the number of training examples m
2. Randomly pick K training examples
3. Set u1,..., uk equal to these K examples
i.e. Pick k distinct random integers i1, i2, ..., ik from 1 to m, and then set u1 = x(i1), u2=x(i2),..., uk=x(ik)

- k-means can be converging to different solutions depending on exactly how the clusters were initialized depending on the random initialization
- k-means can end up at local optima
- k-means can get stuck at different local optima, to avoid this we can try multiple random initializations

we perform random initialization as follows:
```
for i = 1 to 100 {
    randomly initialize k-means
    run k-means. Get c(1), ..., c(m), u1,..., uk
    compute cost function(distortion)
    J(c(1), ..., c(m), u1,..., uk)
}
```
finally we pick one that gives us the lowest cost, it seems that if you are running k-means with small no.of clusters(like 2-10) can make sure to find a 
better local optima, make sure you find better clustering data.
<a name="k_means_choosing_number_of_clusters"></a>
### Choosing the number of clusters
How to choose the value of parameter K
1. The elbow-method
2. if we are running for later downstream purposes, we will evaluate k-means based on a metric for how well it performs for that later purpose
to summarize for the most part the number of clusters (K) is still chosen by hand or human insight, always ask the question *what purpose are you running k-means for?*
and then to think what is the no.of clusters K, that serves the purpose

## Dimensionality Reduction
<a name="dimensionality_reduction_data_compression"></a>
### Motivation 1: Data Compression
Reduce data from 2D to 1D
Suppose we apply dimensionality reduction to a dataset of m examples {x(1), x(2), ..., x(m)}, where x(i) ∈ Rn, as a result of this we get out
A lower dimensional dataset {z(1), z(2),..., z(m)} of m examples where z(i) ∈ Rk for some value of k and k <= n
<a name="dimensionality_reduction_data_visualization"></a>
### Motivation 1: Data Visualization
It is better to visualize data in 2d or 3d, and make more sense of it once dimensionality reduction has been done
<a name="pca"></a>
## Principal Component Analysis
Reduce form 2D to 1D: find a direction (a vector u(1) ∈ Rn) onto which to project the data so as to minimize the projection error
Reduce from n-dimension to k-dimension: Find k vectors u(1), u(2),..., u(k) onto  which to project the data, so as to minimize the projection error

### Algorithm:
Before applying PCA there is a pre-processing step that should be performed
training set: x(1), x(2), ..., x(m)
### Preprocessing(feature scaling/normalization)
uj = 1/m ∑i=1 to m xj(i)
Replace each xj(i) with xj - uj
If different features on different scales(e.g., x1 = size of house, x2 = no.of bedrooms), scale features to have comparable range of values
### Principal Component Analysis Algorithm:
Reduce data from n-dimensions to k-dimensions
Compute "covariance matrix":
 `Sigma(∑) = 1/m ∑ i = 1 to m (x(i))(x(i))T`
Compute "eigen vectors" of matrix ∑:
 `[U, S, V] = svd(Sigma)`, we get
  U = [u(1) u(2) ... u(n)] ∈ Rn * n
We use the first `k` columns from the matrix U to get u(1) to u(k)
Now we wold like to come up with a way to go from your original dataset x ∈ Rn and find a lower dimensional representation z ∈ Rk.
Take first k colums of the U matrix which is a n * k matrix, call it Ureduce.
And the way we compose `z` is 
`z = Ureduce^T * x`, UreduceT will be a k * n matrix, and x can be your training set, cross-validation set of n * 1 matrix, hence z will be a k * 1 matrix
### PCA summary
After mean normalization(ensure every feature has 0 mean) and optionally feature scaling:
   `Sigma(∑) = 1/m ∑ i = 1 to m (x(i))(x(i))T`
   `[U, S, V] = svd(Sigma)` (svd = singular value decomposition)
   `Ureduce = U(:, 1:k)`
   `z = Ureduce' * x`

## Applying PCA 
<a name="pca_reconstruction_from_compressed_representation"></a>  
### Reconstruction from compressed Representation
suppose we want to go from a compresses representation back to our original representation of data, i.e.
we have z = UreduceT x, we want xapprox ∈ Rn which is given by,
`Xapprox = Ureduce . z`, where Ureduce is a n * k matrix, and z is a k * 1 matrix, so Xapprox will be a n * 1 matrix
<a name="pca_choosing_number_of_principal_components"></a>
### Choosing the number of principal components
average squared projection error: `1/m ∑ i = 1 to m || x(i) - xapprox(i) ||^2`
total variation in the data: `1/m ∑ i = 1 to m || x(i) ||^2`
choose the value of k, choose the smallest value so that 
avg. squared error / total variation <= 0.01 (1%)
Algorithm:
 try PCA with k = 1
 compute Ureduce, z(1), z(2), ..., z(m), xapprox(1), ..., xapprox(m)
 check if avg. squared error / total variation <= 0.01
 
if the value of k is not <=0.01 you increment the value of k by 1 and re-run the entire algorithm 

Or using SVD
[U, S, V] = svd(Sigma)
the S matrix is a square matrix, whose diagonal elements are non-zero
we now compute k as follows
`1 - ∑ i = 1 to k Sii / ∑ i = 1 to n Sii` <= 0.01 or,
`∑ i = 1 to k Sii / ∑ i = 1 to n Sii` >= 0.99
in this algorithm you just run svd once, and check for value of k

Note: we said that PCA chooses direction u(1)(or k directions u(1),..., u(k)) onto which to project the data so as to minimize the (squared) projection error.
Another way to say is that PCA tries to minimize:
`1/m ∑ i = 1 to m || x(i) - xapprox(i) ||^2` 

### Advice for applying PCA
Supervised learning speedup
(x(1), y(1)), (x(2), y(2)), ..., (x(m), y(m))
extract inputs:
 unlabeled dataset: x(1), x(2), ..., x(m) ∈ Rn where n can be very large for ex. 10000
 we apply PCA to generate reduced dimension dataset we will call them as 
 z(1), z(2), ..., z(m) ∈ R1000 which will be our new training set, 
 (z(1), y(1)), (z(2), y(2)), ..., (z(m), y(m))
Note: only apply PCA on your training set
### Applications of PCA
- compression
    - reduce memory/disk needed to store data
    - speed up learning algorithm
for the above 2 cases we choose k by % of variance retain    
- Visualization
for this case we choose k = 2 or k = 3
### Bad use of PCA: To prevent overfitting
Use z(i) instead of x(i) to reduce the number of features to k < n, thus meaning less features less likely to overfit
this is actually a bad approach and is not recommended, instead use regularization
`minΘ 1/2m [ ∑i=1 to m (hΘ(x(i)) - y(i))^2 ] + λ/2m ∑j=1 to m Θj^2`
The reason this is not recommended is that if you observe PCA does not use the labels `y` since you only use x(i) to find a lower dimensional approximation of your data
i.e. it throws away a piece of information of your data or reduces the dimension of data. There can be a value in that dimension
Before using PCA, instead of reducing the dimension of data, train your learning algorithm on your original data. Only if your algorithm is slow, requires large amount of data
or only using x(i) doesn't work, then only go for PCA.
So, recommended/good applications of PCA:

 - to compress data so it takes up less computer memory/disk space
 - to reduce the dimension of the input data so as to speed up a learning algorithm
 - to visualize high-dimensional data(by choosing k = 2 or k = 3)
      
# Density Estimation
## Anomaly Detection
Given a dataset {x(1), x(2),..., x(m)} assuming that these examples are non-anamolous
We want an algorithm that tells us whether xtest is anomalous
Given the above training set, we are going to build a model p(x), i.e. probability of x, so
 if p(xtest) < epsilon(ϵ) -> we flag this as an anomaly, also p(xtest) > ϵ -> OK
examples:
 - fraud detection: identify unusual users by checking which have p(x) < ϵ
 - manufacturing
 - monitoring computers in a datacenter
Note: If our anomaly detector is flagging too many anomalous examples, then we need to decrease our threshold ϵ
<a name="anomaly_detection_gaussian_distribution"></a>      
### Gaussian distribution
The Gaussian Distribution is a familiar bell-shaped curve that can be described by a function `N(μ,σ^2)`
Let `x ∈ ℝ`. If the probability distribution of x is Gaussian with mean `μ`, variance `σ^2`, then:

`x ∼ N(μ,σ2)`
The little `∼` or 'tilde' can be read as "distributed as."
The Gaussian Distribution is parameterized by a mean and a variance.

Mu, or μ, describes the center of the curve, called the mean. The width of the curve is described by sigma, or σ, called the standard deviation.

The full function is as follows:

`p(x;μ,σ2) = 1/σ(√2π)e^−1/2 * (x−μ/σ)^2`

### Parameter estimation
Given a dataset, we want to estimate what are the values of `μ` and `σ^2`
We can estimate the parameter `μ` from a given dataset by simply taking the average of all the examples:

`μ = 1/m ∑i = 1 to m x(i)`
We can estimate the other parameter, σ2, with our familiar squared error formula:

`σ^2 = 1/m ∑i = 1 to m (x(i)−μ)^2`
<a name="anomaly_detection_algorithm"></a>
### Algorithm
Given a training set of examples, {x(1),…,x(m)} where each example is a vector, x∈ℝn.

p(x)=p(x1;μ1,σ21)p(x2;μ2,σ22)⋯p(xn;μn,σ2n)
In statistics, this is called an "independence assumption" on the values of the features inside training example x.

More compactly, the above expression can be written as follows:

=`∏j=1 to n p(xj;μj,σ2j)`

1. Choose features xi that you think might be indicative of anomalous examples. Fit parameters μ1,…,μn,σ21,…,σ2n
   
2. Calculate μj = 1/m ∑i = 1 to m xj(i)
   Calculate σj^2 = 1/m ∑i = 1 to m (x(i)j−μj)^2   

3. Given a new example x, compute p(x):
   
   `p(x)=∏ j = 1 to n p(xj;μj,σj^2)=∏j = 1 to n 1/√2πσj exp(−(xj−μj)^2/2σj^2)` Anomaly if p(x)<ϵ

A vectorized version of the calculation for μ is `μ = 1/m ∑i=1 to m x(i)`. You can vectorize σ2 similarly.
<a name="anomaly_detection_developing"></a>
### Developing and evaluating an anomaly detection system
To evaluate our learning algorithm, we take some labeled data, categorized into anomalous and non-anomalous examples ( y = 0 if normal, y = 1 if anomalous).
Among that data, take a large proportion of good, non-anomalous data for the training set on which to train p(x).

Then, take a smaller proportion of mixed anomalous and non-anomalous examples (you will usually have many more non-anomalous examples) for your cross-validation and test sets.

For example, we may have a set where 0.2% of the data is anomalous. 
We take 60% of those examples, all of which are good (y=0) for the training set. 
We then take 20% of the examples for the cross-validation set (with 0.1% of the anomalous examples) and another 20% from the test set (with another 0.1% of the anomalous).
In other words, we split the data 60/20/20 training/CV/test and then split the anomalous examples 50/50 between the CV and test sets.

**Algorithm Evaluation**
Fit model p(x) on training set {x(1),…,x(m)}
On a cross validation/test example x, predict:

If p(x) < ϵ (anomaly), then y=1

If p(x) ≥ ϵ (normal), then y=0

Possible evaluation metrics (see "Machine Learning System Design" section):

True positive, false positive, false negative, true negative.
Precision/recall
F1 score
Note that we use the *cross-validation set to choose parameter ϵ*

Q: Is classification accuracy a good way to measure the algorithm's performance?
A: *No, because of skewed classes(since an algorithm that always predicts y=0 will have high accuracy)*
<a name="anomaly_detection_vs_supervisedlearning"></a> 
### Anomaly detection vs. Supervised Learning
When do we use anomaly detection and when do we use supervised learning?

Use anomaly detection:

We have a very small number of positive examples (y=1 ... 0-20 examples is common) and a large number of negative (y=0) examples.
We have many different "types" of anomalies and it is hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far.

Use supervised learning:

We have a large number of both positive and negative examples. In other words, the training set is more evenly divided into classes.
We have enough positive examples for the algorithm to get a sense of what new positives examples look like. The future positive examples are likely similar to the ones in the training set.
<a name="anomaly_detection_features_to_use"></a>
### Choosing What Features to Use
Suppose your anomaly detection algorithm is performing poorly and outputs a large value of p(x) for many normal examples and for many anomalous examples in your
C.V dataset, then you would try coming up with more features to distinguish between the normal and the anomalous examples.
The features will greatly affect how well your anomaly detection algorithm works.

We can check that our features are gaussian by plotting a histogram of our data and checking for the bell-shaped curve.

Some transforms we can try on an example feature x that does not have the bell-shaped curve are:

log(x)
log(x+1)
log(x+c) for some constant
x√
x1/3
We can play with each of these to try and achieve the gaussian shape in our data.

There is an error analysis procedure for anomaly detection that is very similar to the one in supervised learning.

Our goal is for p(x) to be large for normal examples and small for anomalous examples.

One common problem is when p(x) is similar for both types of examples. 
In this case, you need to examine the anomalous examples that are giving high probability in detail and try to figure out new features that will better distinguish the data.

In general, choose features that might take on unusually large or small values in the event of an anomaly.

## Recommender Systems
<a name="recommender_system_problem_formulation"></a> 
### Problem Formulation
Recommendation is currently a very popular application of machine learning.

Say we are trying to recommend movies to customers. We can use the following definitions

nu= number of users
nm= number of movies
r(i,j)=1 if user j has rated movie i
y(i,j)= rating given by user j to movie i (defined only if r(i,j)=1)
<a name="recommender_system_content_based_recommendation"></a>
### Content Based Recommendations
We can introduce two features, x1 and x2 which represents how much romance or how much action a movie may have (on a scale of 0−1).

One approach is that we could do linear regression for every single user. For each user j, learn a parameter θ(j)∈ℝ3. Predict user j as rating movie i with (θ(j))Tx(i) stars.

θ(j)= parameter vector for user j
x(i)= feature vector for movie i
For user j, movie i, predicted rating: (θ(j))T(x(i))
m(j)= number of movies rated by user j
To learn θ(j), we do the following

`minθ(j)=1/2 ∑i:r(i,j) = 1 ((θ(j))T(x(i)) − y(i,j))^2 + λ/2 ∑k = 1 to n (θ(j)k)^2`
This is our familiar linear regression. The base of the first summation is choosing all i such that r(i,j)=1.

To get the parameters for all our users, we do the following:
minθ(1),…,θ(nu) = `1/2 ∑j = 1 to nu ∑i:r(i,j) = 1 ((θ(j))T(x(i)) − y(i,j))^2 + λ/2 ∑j=1 to nu ∑k=1 to n (θ(j)k)^2`

We can apply our linear regression gradient descent update using the above cost function.

The only real difference is that we eliminate the constant 1m.
<a name="collaborative_filtering"></a> 
## Collaborative Filtering
It can be very difficult to find features such as "amount of romance" or "amount of action" in a movie. To figure this out, we can use feature finders.

We can let the users tell us how much they like the different genres, providing their parameter vector immediately for us.

To infer the features from given parameters, we use the squared error function with regularization over all the users:

`minx(1),…,x(nm)1/2 ∑i = 1 to nm ∑j:r(i,j)=1 ((θ(j))Tx(i) − y(i,j))^2 + λ/2 ∑i = 1 to nm ∑k = 1 to n (x(i)k)^2`
You can also **randomly** guess the values for theta to guess the features repeatedly. 
You will actually converge to a good set of features.

### Collaborative Filtering Algorithm
To speed things up, we can simultaneously minimize our features and our parameters:

`J(x,θ)=1/2 ∑(i,j):r(i,j) = 1 ((θ(j))Tx(i) − y(i,j))^2 + λ/2 ∑ i = 1 to nm ∑k = 1 to n (x(i)k)^ 2 + λ/2 ∑j=1 to nu ∑k=1 to n(θ(j)k)^2`
We have combined the cost function for theta and the cost function for x.

Because the algorithm can learn them itself, the bias units where x0=1 have been removed, therefore x∈ℝn and θ∈ℝn.

These are the steps in the algorithm:

Initialize x(i),...,x(nm),θ(1),...,θ(nu) to small random values. This serves to break symmetry and ensures that the algorithm learns features x(i),...,x(nm) that are different from each other.
Minimize J(x(i),...,x(nm),θ(1),...,θ(nu)) using gradient descent (or an advanced optimization algorithm).
E.g. for every j=1,...,nu,i=1,...nm:
`x(i)k:=x(i)k−α( ∑j:r(i,j) = 1 ((θ(j))Tx(i) − y(i,j))θ(j)k + λx(i)k)`
`θ(j)k:=θ(j)k−α( ∑i:r(i,j) = 1 ((θ(j))Tx(i) − y(i,j))x(i)k + λθ(j)k)`
For a user with parameters θ and a movie with (learned) features x, predict a star rating of θTx.   
<a name="low_rank_matrix_factorization"></a>
### Vectorization: Low Rank Matrix Factorization
Given matrices X (each row containing features of a particular movie) and Θ (each row containing the weights for those features for a given user), then the full matrix Y of all predicted ratings of all movies by all users is given simply by: `Y=XΘT`.

Predicting how similar two movies i and j are can be done using the distance between their respective feature vectors x. 
Specifically, we are looking for a small value of `||x(i)−x(j)||`.
<a name="mean_normalization"></a>
### Implementation Detail: Mean Normalization
If the ranking system for movies is used from the previous lectures, then new users (who have watched no movies), will be assigned new movies incorrectly. Specifically, they will be assigned θ with all components equal to zero due to the minimization of the regularization term. That is, we assume that the new user will rank all movies 0, which does not seem intuitively correct.

We rectify this problem by normalizing the data relative to the mean. First, we use a matrix Y to store the data from previous ratings, where the ith row of Y is the ratings for the ith movie and the jth column corresponds to the ratings for the jth user.

We can now define a vector
μ=[μ1,μ2,…,μnm]
such that

μi=∑j:r(i,j)=1Yi,j / ∑jr(i,j)
Which is effectively the mean of the previous ratings for the ith movie (where only movies that have been watched by users are counted). We now can normalize the data by subtracting u, the mean rating, from the actual ratings for each user (column in matrix Y):

As an example, consider the following matrix Y and mean ratings μ:
Y = [5 5 0 0
     4 ? ? 0
     0 0 5 4
     0 0 5 0]
μ=[2.5
   2
   2.25
   1.25]

Y' = [2.5 2.5 -2.5 -2.5
      2   ?    ?   -2
     -2.25 -2.25 3.75 1.25
     -1.25 -1.25 3.75 -1.25]   

Now we must slightly modify the linear regression prediction to include the mean normalization term:

`(θ(j))Tx(i) + μi`
Now, for a new user, the initial predicted values will be equal to the μ term instead of simply being initialized to zero, which is more accurate.
<a name="large_datasets">
## Learning with Large Datasets
We mainly benefit from a very large dataset when our algorithm has high variance when m is small. 
Recall that if our algorithm has high bias, more data will not have any benefit.

Datasets can often approach such sizes as m = 100,000,000. In this case, our gradient descent step will have to make a summation over all one hundred million examples. We will want to try to avoid this -- the approaches for doing so are described below.
<a name="stochastic_gradient_descent"></a> 
### Stochastic Gradient Descent
Stochastic gradient descent is an alternative to classic (or batch) gradient descent and is more efficient and scalable to large data sets.

Stochastic gradient descent is written out in a different but similar way:
`cost(θ,(x(i),y(i))) = 1/2 (hθ(x(i)) − y(i))^2`
The only difference in the above cost function is the elimination of the m constant within 1/2
`Jtrain(θ) = 1/m ∑i = 1 to m cost(θ,(x(i), y(i)))`
Jtrain is now just the average of the cost applied to all of our training examples.
The algorithm is as follows

 - Randomly 'shuffle' the dataset
 - For i=1…m
 `Θj: = Θj − α(hΘ(x(i)) − y(i))⋅x(i)j`
This algorithm will only try to fit one training example at a time. 
This way we can make progress in gradient descent without having to scan all `m` training examples first. 
When the training size `m` is very large stochastic gradient descent can be much faster 
Stochastic gradient descent will be unlikely to converge at the global minimum and will instead wander around it randomly, but usually yields a result that is close enough. 
Stochastic gradient descent will usually take 1-10 passes through your data set to get near the global minimum.
The cost function should go down with every iteration of batch gradient descent but not necessarily with stochastic gradient descent
<a name="minibatch_gradient_descent"></a> 
### Mini-Batch Gradient Descent
Mini-batch gradient descent can sometimes be even faster than stochastic gradient descent. 
Instead of using all m examples as in batch gradient descent, and instead of using only 1 example as in stochastic gradient descent, we will use some in-between number of examples b.

Typical values for b range from 2-100 or so.

For example, with b=10 and m=1000:
For i=1,11,21,31,…,991
`θj:=θj − α 1/10 ∑k=i to i+9 (hθ(x(k)) − y(k))x(k)j`
We're simply summing over ten examples at a time. 
The advantage of computing more than one example at a time is that we can use vectorized implementations over the b examples.
<a name="stochastic_gradient_descent_convergence"></a>
### Stochastic Gradient Descent Convergence
How do we choose the learning rate α for stochastic gradient descent? Also, how do we debug stochastic gradient descent to make sure it is getting as close as possible to the global optimum?

One way is to plot the average cost of the hypothesis applied to every 1000 or so training examples. 
We can compute and save these costs during the gradient descent iterations.

With a smaller learning rate, it is possible that you may get a slightly better solution with stochastic gradient descent. 
That is because stochastic gradient descent will oscillate and jump around the global minimum, and it will make smaller random jumps with a smaller learning rate.

If you increase the number of examples you average over to plot the performance of your algorithm, the plot's line will become smoother.

With a very small number of examples for the average, the line will be too noisy and it will be difficult to find the trend.

One strategy for trying to actually converge at the global minimum is to **slowly decrease α** over time. For example
 `α=const1/iterationNumber + const2`

However, this is not often done because people don't want to have to fiddle with even more parameters.
<a name="online_learning"></a>
### Online Learning
With a continuous stream of users to a website, we can run an endless loop that gets (x,y), where we collect some user actions for the features in x to predict some behavior y.

You can update θ for each individual (x,y) pair as you collect them. 
This way, you can adapt to new pools of users, since you are continuously updating theta.
<a name="mapreduce_data_parallelism"></a>
### Map Reduce and Data Parallelism
We can divide up batch gradient descent and dispatch the cost function for a subset of the data to many different machines so that we can train our algorithm in parallel.
You can split your training set into z subsets corresponding to the number of machines you have. On each of those machines calculate
∑i=p to q(hθ(x(i)) − y(i))⋅x(i)j, where we've split the data starting at p and ending at q.
MapReduce will take all these dispatched (or 'mapped') jobs and 'reduce' them by calculating:

Θj:=Θj−α1/z(temp(1)j + temp(2)j+⋯+temp(z)j)
For all j=0,…,n.

This is simply taking the computed cost from all the machines, calculating their average, multiplying by the learning rate, and updating theta.

Your learning algorithm is MapReduceable if it can be expressed as computing sums of functions over the training set. Linear regression and logistic regression are easily parallelizable.

For neural networks, you can compute forward propagation and back propagation on subsets of your data on many machines. Those machines can report their derivatives back to a 'master' server that will combine them.