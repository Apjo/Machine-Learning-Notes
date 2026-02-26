# Calculus & Derivatives

## Derivative
- instantaneuous rate of change
- The derivative of a function at a point is precisely the slope of the tangent at that particular point.

### Zero slope
- a horizontal line has a slope of 0
maxima/minima
- a place where derivative of a function is 0

Notes

`y1=ax1+b`, `y2=cx1+d` then 
`slope of y` depends on `a`, independent of `b` since `b` and `d` are constants, and if `a > c` then `slope of y1 > slope of y2`

### Slope of a Line

The slope of a line measures its steepness and direction. It is defined as the ratio of the vertical change (rise) to the horizontal change (run) between two points on the line.

Calculation of Slope:

To calculate the slope (m) between two points, (x₁, y₁) and (x₂, y₂), you can use the formula:

`m ={y₂ - y₁}/{x₂ - x₁}` 

Where:
* `(y₂ - y₁)` is the change in the y-coordinates (rise).
* `(x₂ - x₁)` is the change in the x-coordinates (run).

### Nuances to Keep in Mind
* `Positive Slope`  : Indicates the line rises as it moves from left to right.
* `Negative Slope`  : Indicates the line falls as it moves from left to right.
* `Zero Slope`      : A horizontal line has a slope of 0, meaning there is no vertical change.
* `Undefined Slope` : A vertical line has an undefined slope because the run (denominator) is 0.

Understanding these concepts will help you analyze linear relationships effectively.
- slope at a point = `dx/dt`
- point slope=`y-y1 = m(x-x1)`
- Given `y=mx+b`, `b` is the `y`-intercept, and `m`=slope
- Recap on lines:
    - when we have `y`=`C`(constant) then we know we have 1 line at the y-intercept, and thats a horizontal line(or x=0 always)
    - Similarly, when we have `x`=`C`,then we know we have 1 line at the X-intercept, and thats a vertical line(or y=0 always) 
    - Bring the equation `4x+2y-3`=0 in slop intercept form, so
        - we just simplify to get `y =3-4x/2`
    - Now, to get x and y intercepts, we simply make x and y to be 0, and derive values for x and y
    - Parallel lines:
        - have same slope
    - Perpendicular lines:
        - meet at 90 degrees
        - slopes are negative reciprocals of each other 

Example Problem:

Q: Find the eqtn of line passing through point(6,7) parallel to the line `2x+3y=12`

First we bring the equation in the y-intercept formula `y=12-2x/3 = 4 - 2x/3`

Then we know we want a parallel line, so the slope `m = -2/3` is the same, and now we use the point slope formula, and plug in (6,7) as

```
y-y1 = m(x-x1)
y-7 = -2/3(x-6)
y-7 = -2x/3+4
y = -2x/3 + 11
```

### Angle of Inclination
- TBC

### Lagrange's and Leibniz's notation

say we have a function y = f(x)

Derivative of f is expressed as
`f'(x)` (or f prime) in Lagrange's notation == slope
and `dy/dx = d/dx f(x)` in Leibniz's notation

## Derivatives

- derivative of a constant is always 0
- if f(x) = a(x) + b, f'(x) = a
- if y=f(x)=x^2, f'(x) = 2x
- y=f(x)=x^3, f'(x) = 3x^2
- y=f(x)=1/x = -x^-2
- so, if f(x)  = x^n, then f'(x) = n * x^n-1
- if f and g are inverse functions then g'(y) = 1/f'(x)
- y=f(x) = sin(x), f'(x) = cos(x)
- y=f(x) = cos(x), f'(x) = -sin(x)
- eulers number
    - e=2.718281828...
    - f(x) = e^x, then f'(x) = e^x
- Derivative of log(x)
    - Say we have f(x)=e^x, An inverse function undoes the action of the original function
    - To find the inverse of ( f(x) = e^x ), we set ( y = e^x ).
        * We want to express ( x ) in terms of ( y ): [ y = e^x ]
        * Taking the natural logarithm (logarithm base ( e )) of both sides gives: [log(y) = x ]
        * From the equation ( log(y) = x ), we can express the inverse function: [ f^{-1}(y) = log(y) ]
        * Therefore, the inverse of the exponential function ( f(x) = e^x ) is the logarithmic function: [ f^{-1}(y) = log(y) ]
    -   say y=e^x, which follows after taking natural logs on both sides log(y) = x
    - then f^-1(y) = log(y), also note e^log(x) =x, and hence log(e^y) = y
    * This derivation shows how the logarithmic function is defined as the inverse of the exponential function

## Non differentiable functions

* A non-continous function is not differentiable, also derivative of mod(x) is not possible, basically a cusp or a corner
* Any function that doesn't have continuity is not differentiable
* A function with a vertical tangent

## Multiplication by Scalars

* When a function ( f ) is multiplied by a constant ( c ), the derivative ( f' ) is equal to ( c ) times the derivative of the function being multiplied.
*  if `( f = 2g )`, then `( f' = 2g' )`.

## Properties of a derivative

### Sum Rule

if `f = g + h`, then `f' = g' + h'`, so if `f(x)=2x`, `g(x)=x^2`, then `(f+g)'(x) = 2x+2`

### Product Rule

- Given `f=gh`, then `f' = g'h + gh'`, so if `f(x)=x.e^x`, then `f'(x) = e^x+xe^x`

### Chain Rule

Chain rule states that the derivative of composite function `f(g(x))` is `f'(g(x))⋅ g'(x)`. 
* In other words, `cos(4x)`, is a composite function and it can be written as `f(g(x))` where `f(x) = cos(x)` and `g(x) = 4x`.
* We can then compute the derivative of `cos(4x)` using the chain rule and the derivatives of `cos(x)` and `4x`.
* `d/dx ( f(g(x) ) = f' (g(x)) · g' (x)`
* `dy/dx = dy/du · du/dx`

## Introduction to Optimization

If you want to optimize a function whether maximizing it or minimizing it & the function is differentiable at every point then candidates for maximum, and minimum are those points for which the derivative is zero.

### Optimization of squared loss

minimize `(x-a1)^2 + (x-a2)^2 + ...+(x-an)^2` then solution
`x = a1 + a2 + a3+...+an / n`

### Optimization of log loss

Why do we need it?
Log loss, also known as logistic loss or cross-entropy loss, is significant in machine learning for several reasons:
* `Performance Measurement`: Log loss quantifies the performance of a classification model whose output is a probability value between 0 and 1. It measures how well the predicted probabilities align with the actual class labels.
* `Sensitivity to Confidence`: Unlike other loss functions, log loss penalizes incorrect predictions more heavily when the model is confident but wrong. For example, predicting a probability of 0.9 for a class that is actually 0 will incur a higher penalty than predicting 0.6.
* `Optimization`: Log loss is differentiable, which makes it suitable for optimization algorithms like gradient descent. This allows models to learn effectively by adjusting weights to minimize the log loss during training.
* `Probabilistic Interpretation`: Log loss provides a probabilistic interpretation of the model's predictions, making it useful for applications where understanding uncertainty is important, such as in medical diagnoses or risk assessment.
* `Common in Classification Tasks`: It is widely used in binary and multi-class classification problems, especially in logistic regression and neural networks.

Maximizing the function `(g(p))`, which represents the probability of winning in our coin toss game, can be quite complex. However, using the logarithm of `( g(p) )` simplifies the process significantly. The logarithm has a special property: it transforms products into sums. This means that instead of dealing with the multiplication of probabilities, we can work with the addition of their logarithms, which is much easier to handle mathematically.
For example, if we have `g(p) = p^7 *(1 - p)^3`, taking the logarithm gives us:
`log(g(p)) = log(p^7) + log((1 - p)^3) = 7*log(p) + 3 *log(1 - p)`

This transformation allows us to use simpler calculus techniques to find the maximum value. Since maximizing `( g(p) )` and maximizing `( log(g(p)) )` yield the same result, we can focus on the easier expression. This is a common technique in machine learning, especially when dealing with probabilities, as it often leads to more manageable calculations.

If log loss is minimized too much, it can lead to a few potential issues:
* `Overfitting`: The model may become overly complex and tailored to the training data, capturing noise rather than the underlying patterns. This results in poor generalization to unseen data, leading to high error rates on validation or test sets.
* `Confidence in Wrong Predictions`: A model that minimizes log loss excessively might produce very confident predictions (probabilities close to 0 or 1) for incorrect classifications. This can be problematic, especially in applications where understanding uncertainty is crucial.
* `Loss of Robustness`: The model may become sensitive to small changes in the input data, making it less robust. This can lead to significant performance drops when the model encounters slightly different data distributions.
* `Diminishing Returns`: After a certain point, further minimizing log loss may yield diminishing returns in terms of actual performance improvement. The focus should be on achieving a balance between minimizing log loss and maintaining model simplicity and generalization.
To mitigate these issues, techniques such as cross-validation, regularization, and monitoring performance on validation datasets are often employed

While calculating the log loss we use `-G(p)`, as the log returns a negative number when `p` is between `0` and `1` so we want `-G(p)` to be a positive number, and instead of maximizing `G(p)`we minimize `-G(p)`

## Functions with 2 or more variables

### Tangent plane

Contains tangent lines from the given number of variables i.e. `f(x,y)= x^2 + y^2` will contain 2 tangent lines, one for `x`, and for `y`. This gives rise to partial derivatives

### Partial Derivatives

If we have a equation with say 2 variables, `f(x,y)= x^2 + y^2`, and we wanted to take derivatives, we do it in 2 steps, i.e. 
in one step we take `y` as a constant hence we calculate. derivate of `f(x)` over `x` i.e. `f'(x)`, or as we call it "partial derivative of f wrt x"
and in the next step, we take `x` as a constant so we calculate derivate of `f` over `y` i.e. `f'(y)` or as we call it "partial derivative of f wrt y"

so, `f'(x) = 2x`, and `f'(y) = 2y`

### Basic steps
So, the basic steps are:
1. Treat all other variables as constant.
2. Differentiate the function using the normal rules of differentiation


## Gradients & Gradient descent
if `f(x,y) = x^2 + y^2`, then gradient of `f = [  2x
 										         2y ]`
basically the **gradient** of `f(x,y)` will be a *vector of its partial derivatives*. The gradient points to the steepest ascent because it is derived from the rates of change of the function in all directions, and it directs you to the path where the function increases the most. When you calculate the gradient of a function `(f(x, y))`, you get a vector that points in the direction of the greatest increase of the function. 

Mathematically, if you take a small step in the direction of the gradient, the increase in the function's value is maximized. This is due to the properties of derivatives, where the rate of change is highest in the direction of the gradient.

Works only on differentiable functions, since the **Gradient Descent** uses the **Gradient** as its base, and the gradient is related to partial derivatives, we *must have differentiable functions to perform the algorithm*.

Remember `gradient descent` is a method used to find the _lowest point_ on a hill, or the minimum of a function. Instead of climbing up, you're trying to go down. Think of it as taking small steps downhill based on the direction indicated by the gradient. Each step you take gets you closer to the lowest point. **We need gradient descent when we want to minimize a cost function**, which measures how well our model is performing. 

By using gradient descent, we can adjust the parameters of our model to improve its accuracy.

### You use it when:
* You have a function you want to minimize, like a **loss function** in regression or classification.
* The function is too complex to solve analytically (e.g., no closed-form solution).
* You want to learn parameters (like **weights** in linear regression or neural networks) that make your model perform better.

### How it works?
1. Start with initial guesses for the parameters (e.g., weights).
2. Compute the gradient — the direction of steepest ascent.
3. Move in the opposite direction of the gradient (i.e., downhill).
4. Repeat until the changes become very small (i.e., you reach a minimum).

The update rule for a parameter
`θ(theta) := θ − α⋅∂E/∂θ`

Where:
* `α(alpha)` (learning rate): step size.
* `∂E/∂θ` (partial derivative of E with respect to θ): the gradient of the cost function with respect to `θ`

## Gradients and maxima/minima

### Optimization using Gradient Descent in one variable

To find the minimums and maximums, set all the partial derivatives to be equal to `0`, and solve that system of equations

**Algo to find the minimum of `f(x)`:**

**Goal: Find minimum of `f(x)`**

**Step1:**

Define a learning rate `α(alpha)`, here The learning rate ensures that the steps we are performing are small enough so the algorithm can converge to the minimum.

Chose starting point `x0`

**Step2:**

Update `xk = xk-1 * alpha f'(xk-1)`

**Step3:**
repeat step2 until you are close enough to the true minimum `x*`

NOTE: We also need `num_iterations` as a parameter as the number of iterations tells us how many times we will perform the calculations. Higher number of iterations will lead to more precise results but will take longer to perform the computations.

## Implement a simple gradient descent

### Here your `dfdx` is to be implemented and depends on what `x` is, so lets say our `f(x) = e^x - log(x)`
```python

def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

def simple_gradient_descent(dfdx,x, num_iteration, learning_rate):
	for iteration in range(num_iterations):
		x = x - learning_rate*dfdx(x)
return x
```

### Optimization using Gradient Descent in two variables

**Function:** `f(x,y)`

**Goal:** Find minimum of `f(x,y)`

**Step1:**

Define a learning rate `α(alpha)`, & chose starting point `(x0,y0)`

**Step2:**
Update 
```
         [xk      [xk-1  
             =           - alpha * f'(xk-1, yk-1)
         yk]       yk-1]
```
**Step3:**
Repeat step 2 until you are close enough to the true minimum `(x*, y*)`

### Optimization using Gradient Descent - Least Squares with multiple observations

### Regression with a perceptron

A perceptron can be seen as a model for linear regression, where the goal is to predict house prices based on features like size and number of rooms. For example, if you want to predict the price of a house based on its size and number of rooms, you can use a perceptron to do that.

The perceptron takes multiple inputs (features) and produces an output (predicted price) through a summation function.

### Mathematical Representation

Each input feature is multiplied by a corresponding weight, indicating its importance in predicting the output.
The output is calculated as `y^ = w1*x1 + w2*x2 + b`, where `w1` and `w2` are weights, `x1` and `x2` are input features, and `b` is a bias term.

### Optimization of Weights

The objective is to find the optimal weights and bias that minimize prediction error, which is assessed using a loss function. The loss function measures how far the predicted prices are from the actual prices, guiding the optimization process. The loss function is a measure of how good or bad the model is performing. The lower the loss, the better the model is performing.


#### Single layered Neural Network

**Goal:**

Find weights and bias that will optimize the predictions i.e. reduce the errors in predictions
`y^=w1x1+w2x2+b` is function for predicted values goal is to find optimal values for  `w1`, `w2` and `b` s.t. we minimize the error! so the **predicted value** `y^` will be treated as a function of `w1`, `w2`, and `b`

#### Loss Function

For each training example you can measure the difference between original values `𝑦(𝑖)` and predicted values `y^(𝑖)` with the loss function

`L(y,y^) = 1/2*(y-y^)^2` say this is **EQTN (0)** 
here Division by 2 is taken just for scaling purposes. Our goal now is to minimize this. To find the `w1`, `w2`, and `b`, that gives `y^` with the least error. To compare the resulting vector of the predictions `y^(1×𝑚)` with the vector `Y` of original values `𝑦(𝑖)`, you can take an average of the loss function values for each of the training examples:

`L(w, b) = 1/2m * Sum for all i going from 1 to m of (y-y^)^2`

This function is called the **sum of squares cost function**. The aim is to optimize the cost function during the training, which will minimize the differences between original values `𝑦(𝑖)` and predicted values `𝑦^(𝑖)`.

When your weights were just initialized with some random values, and no training was done yet, you can't expect good results. You need to calculate the adjustments for the weight and bias, minimizing the cost function. This process is called **backward propagation**

```
**EQTN (1)**

w1 -> w1 - alpha(∂L/∂w1)
w2 -> w2 - alpha(∂L/∂w2)
b -> b - alpha(∂L/∂b)
```
Using chain rule on **EQTN (1)** we get:

```
**EQTN (2)**

∂L/∂w1 = ∂L/∂y^ * ∂y^/∂w1
∂L/∂w2 = ∂L/∂y^ * ∂y^/∂w2
∂L/∂b  = ∂L/∂y^ * ∂y^/∂b
```

We also know that `y^ = w1x1 + w2x2 + b` and from `EQTN (0)` and we also know that `L(y,y^) = 1/2*(y-y^)^2`

Get the partial derivative of `L` with respect to `y^` as:

```
**EQTN (3)**
L(y,y^) = 1/2*(y-y^)^2
∂L/∂y^  = d/dy^ (1/2*(y-y^)^2)
        = 1/2*2*(y-y^)* d/dy^ (y-y^) (using chain rule)
        = (y-y^)* d/dy^ (y-y^)
        = 1/2*2*(y-y^)* -1 (using chain rule)
        = -(y-y^) (since d/dy^ (y-y^) = -1)

```

also recollect that `y^ = w1x1 + w2x2 + b` is a function of `w1`, `w2`, and `b`, hence we can write the partial derivatives of `y^` with respect to `w1`, `w2`, and `b` as: 

```
**EQTN (4)**

∂y^/∂w1 = x1
∂y^/∂w2 = x2
∂y^/∂b = 1
```
putting above values from **EQTN (3)** and **EQTN (4)** in **EQTN (2)**, call as : 

```
**EQTN (5)**

∂L/∂w1 = ∂L/∂y^ * ∂y^/∂w1 = -(y-y^)* x1

∂L/∂w2 = ∂L/∂y^ * ∂y^/∂w2 = -(y-y^)* x2

∂L/∂b = ∂L/∂y^ * ∂y^/∂b = ∂L/∂y^ * 1 = ∂L/∂y^ = -(y-y^)

```
Recall our main goal was to find optimal values for `w1`, `w2`, and `b` to give our predictions `y^` with smallest possible error, so rewriting **EQTN (1)** using values from **EQTN (5)** as:

**EQTN (1)** becomes:

```
w1 -> w1 - alpha(∂L/∂w1) =  w1 - alpha(-x1(y-y^)) 
w2 -> w2 - alpha(∂L/∂w2) =  w2 - alpha(-x2(y-y^))
b ->  b  - alpha(∂L/∂b)    =  b - alpha(-(y-y^))
```
This is the gradient descent step. If we do this many times, we will get some really good weights, `w1`, `w2`, and `b`, that are going to have a really small error and therefore a really good model.

### Linear regression using Gradient Descent implementation
```
import numpy as np 

# E=1/2n * sum for i from 1 to n (m*x(i) + b - y(i))^2
def sum_of_squares_cost_function(m, b, X, Y):
    y_pred = m*X + b
    return 1/2*len(Y) * np.sum((Y - y_pred)**2) 

#∂E/∂m = 1/n * sum for i from 1 to n (m*x(i) + b - y(i))*x(i)
def partial_derivative_of_loss_with_respect_to_m(m, b, X, Y):
    y_pred = m*X + b
    return 1/len(Y) * np.dot(y_pred - Y, X)

#∂E/∂b = 1/n * sum for i from 1 to n (m*x(i) + b - y(i))*(1)
def partial_derivative_of_loss_with_respect_to_b(m, b, X, Y):
    y_pred = m*X + b
    return 1/len(Y) * np.sum(y_pred - Y)

def gradient_descent_step(X, Y, m, b, alpha, num_iterations, dEdM, dEdB):
    for epoch in range(num_iterations):
        m_new = m - alpha * dEdM(m, b, X, Y)
        b_new = b - alpha * dEdB(m, b, X, Y)
        m = m_new
        b = b_new
    return m, b

def gradient_descent(X, Y):
    m_initial=0, b_initial=0,learning_rate=0.01, num_iterations=1000
    print(
        gradient_descent_step(
            X=X,
            Y=Y,
            m=m_initial,
            b=b_initial,
            alpha=learning_rate,
            num_iterations=num_iterations,
            partial_derivative_of_loss_with_respect_to_m,
            partial_derivative_of_loss_with_respect_to_b))
```

## Classification with a perceptron

A perceptron is a fundamental building block of neural networks, often likened to a simple decision-making unit. Imagine it as a tiny brain cell that takes in information, processes it, and produces an output. Specifically, a perceptron receives multiple inputs, each associated with a weight that signifies its importance. It then combines these inputs, applies a mathematical function (usually a step function or activation function), and produces a single output, which can be thought of as a yes or no decision.



### Sigmoid function

defined as `sigma(z) = 1/1+e^-z` --> say **EQTN (6)** 
This function is called the **sigmoid** function. It maps the result to a value between 0 and 1, thus the output can be interpreted as a probability.

To find derivative of a  sigmoid function `sigma(z)`:


rewriting `1/1+e^-z = (1+e^-z)^-1`
```
**EQTN (7)**

d/dz(sigma(z))  = d/dz((1+e^-z)^-1)
                = -1 * (1+e^-z)^-2 * (d/dz (1+e^-z))
                = -1 * (1+e^-z)^-2 * (d/dz(1) + d/dz (e^-z))
                = -1 * (1+e^-z)^-2 * (0 + e^-z * d/dz(-z))
                = -1 * (1+e^-z)^-2 * (e^-z) * (-1)
                = (1+e^-z)^-2 * (e^-z)
                = 1/(1+e^-z)^2 *(e^-z)

d/dz(sigma(z))= e^-z / (1+e^-z)^2
```

We now try to add 1, and substract 1 from numerator i.e.

```

= e^-z +1 -1/ (1+e^-z)^2
rewriting +1 and -1 as a single term
= 1+e^-z - 1 / (1+e^-z)^2
= (1+e^-z) /  (1+e^-z)^2 - 1/(1+e^-z)^2
= 1/(1+e^-z) - 1/(1+e^-z)^2
= 1/(1+e^-z) - 1/(1+e^-z) * 1/(1+e^-z)
taking 1/1+e^-z common,
= 1/1+e^-z(1 - 1/(1+e^-z)), say **EQTN (8)**
```

We now substitue **EQTN (6)** into **EQTN (8)**, we get

recap: we know `sigma(z) = 1 / 1 + e^-z` From **EQTN (6)**

Hence, `d/dz(sigma(z))= sigma(z)*(1 - sigma(z))` --> **EQTN (9)**

### Gradient Descent

Continuing from above example we have a prediction function `y^ = sigma(w1*x1 + w2*x2 + b)`, and we have our log loss function i.e.

`L(y,y^) = -y*log(y^) - (1-y)*log(1 - y^)`

in order to find optimal values for `w1`, `w2`, and `b`, we need **Gradient Descent**

```
**EQTN (10)**
`w1 -> w1 - alpha*(∂L / ∂w1)`
`w2 -> w2 - alpha*(∂L / ∂w2)`
`b -> b - alpha*(∂L / ∂b)`
```

So, what we know now is We want to calculate all those derivatives and be able to do a **Gradient Descent** step in order to find the best weights and bias for our dataset.

Say for example we focus on `w1`, The idea is to reduce the function `L(y, ŷ)` by moving around with `W1`. Because if we can find in which direction to move `w1` in order to reduce `L` by a little bit, all we need to do is iterate this step with all the weights many times. Now, there are a lot of variables in between `L` and `w1`.

One of them is `y^` because `y^` affects `L`. We need to find the partial derivative of `L` with respect to `y^`. Now, `y^` is affected by `w1`, so we need to find the partial derivative of `y^` by `w1`. Finally, that tells us how much `L` is affected by `w1`. The missing piece is the partial derivative of `L` with respect to `w1`.

```
**EQTN (11)**

∂L/∂w1 = ∂L/∂y^ * ∂y^/∂w1
∂L/∂w2 = ∂L/∂y^ * ∂y^/∂w2
∂L/∂b  = ∂L/∂y^ * ∂y^/∂b
```

Once again to recap, we are given the **prediction function**:
`y^ = sigma(w1*x1 + w2*x2 + b)`

our **log loss function** is:
`L(y, y^) = -y*log(y^) - (1 - y)*log(1 - y^)`

hence `∂L/∂y^` comes out to be

``` 
**EQTN (12)**
L(y, y^)  = -y*log(y^) - (1 - y)*log(1 - y^)
∂L/∂y^    = ∂L/∂y^ * ∂y^/∂y^
		  = -y / y^ - (1 - y) / (1 - y^)
		  = -(y - y^) / (y^ * (1 - y^))
```
Similarly, looking at how we derived derivative of the sigmoid function from **EQTN (9)**:

`d/dz(sigma(z))= sigma(z)(1 - sigma(z))`

applying partial derivative to `y^` with respect to `w1`, `w2`, and `b` we get:
```
**EQTN (13)**
y^ = sigma(w1*x1 + w2*x2 + b)

∂y^/∂w1 = x1
∂y^/∂w2 = x2
∂y^/∂b = 1
```

Substituting values of **EQTN (12)** into **EQTN (11)**:
```
**EQTN (14)**
∂L/∂w1 = ∂L/∂y^ * ∂y^/∂w1
		= -(y-y^)/y^(1-y^) * y^(1-y^)x1
		= -x1(y - y^)

∂L/∂w2 = ∂L/∂y^ * ∂y^/∂w2
		= -(y-y^)/y^(1-y^) * y^(1-y^)x2
		= -x2(y - y^)

∂L/∂b = ∂L/∂y^ * ∂y^/∂b
        = -(y-y^)/y^(1-y^) * 1
	    = -(y-y^)/y^(1-y^)
        = -(y-y^)
```

Now, we know we were trying to find optimal values for w1, w2, and b, and to help with that we need gradient descent, recap **EQTN (10)**,

```
**EQTN (10)**
`w1 -> w1 - alpha*(∂L / ∂w1)`
`w2 -> w2 - alpha*(∂L / ∂w2)`
`b -> b - alpha*(∂L / ∂b)`
```


we substitute **EQTN (14)** values into  **EQTN (10)** as:

**EQTN (10)** becomes:
```
**EQTN (15)**

w1 = w1 - alpha ∂L/∂w1
   = w1 - alpha(-x1(y - y^))

w2 = w2 - alpha ∂L/∂w2
   = w2 - alpha(-x2(y - y^))

b  = b - alpha ∂L/∂b
   = b - alpha(-(y - y^))
```

## Classification with a Neural Network(NN)

A NN is a bunch of perceptrons organized in layers.
Consdier a NN with 2 inputs, 1 hidden layer with 2 neurons, and 1 output layer with 1 neuron.
The hidden layer neurons are connected to the input layer neurons through weights w11, w12, w21, w22
The bias terms for the hidden layer neurons are b1 and b2.
The output layer neuron is connected to the hidden layer neurons through weights w1 and w2.
The bias term for the output layer neuron is b.

The prediction is made by the output layer neuron. The prediction is a sigmoid function of the weighted sum of the inputs to the output layer neuron plus the bias term.

so 
```
**EQTN (16)**
a1 = sigma(z1)
z1 = w11*x1 + w12*x2 + b1
a2 = sigma(z2)
z2 = w21*x1 + w22*x2 + b2
y^ = sigma(z)
z = w1*a1 + w2*a2 + b
```
So, z is the output of the NN, and since we are dealing with a classification problem, the error function is the log loss function.

`L(y,y^) = -y*log(y^) - (1-y)*log(1 - y^)`
where `y^` is the **`prediction`** of the NN, and `y` is the **`target`** in the training data we want to predict.

## Classification with a Neural Network - Minimizing log-loss

Recall our goal was to find optimal values for `w1`, `w2`, and `b` to give our predictions `y^` with smallest possible error, in other words the partial derivatives tells us exactly in what direction to move each one of the weights, and biases in order to reduce the loss.

```
∂L/∂w11 = ∂z/∂w11 * ∂a1/∂z1 * ∂z/∂a1 * ∂y^/∂z * ∂L/∂y^
        = x1 * a1 * (1 - a1) * w1 * y^*(y-y^) * (-(y - y^) / (y^ * (1 - y^)))
        = x1 * a1 * (1 - a1) * w1 * -(y-y^)
        = -x1 * a1 * (1 - a1) * w1 * (y-y^)
```

Perfor gradient descent with

```
w11 -> w11 - alpha * ∂L/∂w11
    -> w11 - alpha * -x1 * a1 * (1 - a1) * w1 * (y-y^)
    -> w11 + alpha * x1 * a1 * (1 - a1) * w1 * (y-y^)
```

to find optimal values for w11 that gives the least error

Similarly, now consider for the bias term `b1`

```
∂L/∂b1 = ∂z/∂b1 * ∂a1/∂z1          * ∂z/∂a1        * ∂y^/∂z        * ∂L/∂y^
        = 1     * a1*(1 - a1)      *w1             * y^*(1-y^)     * (-(y - y^) / (y^ * (1 - y^)))
        = a1 * (1 - a1) * w1 * -(y-y^)
        = -a1 * (1 - a1) * w1 * (y-y^)
```

Perform gradient descent with 

```
b1 -> b1 - alpha * ∂L/∂b1
    -> b1 - alpha * -a1(1-a1) * w1 * y^(1-y^)
    -> b1 + alpha * a1 * (1 - a1) * w1 * (y-y^)
```

to find optimal values for b1 that gives the least error
Now we continue calculating partial derivatives for the weights and biases of the hidden layer neurons and the output layer neuron.

```
w12 -> w11 + alpha * x2 * w1 * a1(1-a1) * (y-y^)

w21 -> w21 + alpha * x1 * w2 * a2(1-a2) * (y-y^)

w22 -> w22 + alpha * x2 * w2 * a2(1-a2) * (y-y^)

b2 -> b2 + alpha * w2 * a2(1-a2) * (y - y^)
```
That finishes the gradient descent for the weights and biases of the hidden layer neurons, now we continue with the gradient descent for the weights and biases of the output layer neuron.


```
∂L/∂w1 = ∂z/∂w1 * ∂y^/∂z    * ∂L/∂y^
        = a1    * y^*(1-y^) * (-(y - y^) / (y^ * (1 - y^)))
        = a1 * -(y-y^)
        = -a1 * (y-y^)
```
Perform gradient descent with 
```
w1 -> w1 - alpha * ∂L/∂w1
    -> w1 - alpha * -a1 * (y-y^)
    -> w1 + alpha * a1 * (y-y^)
```
to find optimal values for `w1` that gives the least error

Similarly, for `w2`, and bias `b`
```
w2 -> w2 - alpha * ∂L/∂w2
    -> w2 - alpha * -a2 * (y-y^)
    -> w2 + alpha * a2 * (y-y^)

b -> b - alpha * ∂L/∂b
    -> b - alpha * -(y-y^)
    -> b + alpha * (y-y^)
```

### Gradient Descent and Backpropagation
A method used to train models, especially neural networks. It is a way to update the weights and biases of the model to minimize the loss function.

The idea is to calculate the gradient of the loss function with respect to the weights and biases, and then use this gradient to update the weights and biases i.e. it starts from the output layer and finishes at the input layer.

As the name suggests, the backpropagation method iteratively updates the neural network parameters from backwards.

If we didn't use backpropagation in training neural networks, several issues would arise:

`Inefficient Learning`: Without backpropagation, the network wouldn't effectively learn from its mistakes. It would be challenging to determine how to adjust the weights and biases based on the errors made during predictions.

`Slow Convergence`: Training would take much longer, as the network would lack a systematic way to update its parameters. It might rely on random adjustments, leading to a very slow and inefficient learning process.

`Poor Performance`: The model's accuracy would likely suffer. Without proper adjustments to the weights and biases, the network would struggle to minimize the loss function, resulting in suboptimal predictions.

`Inability to Handle Complex Problems`: Backpropagation allows neural networks to learn complex patterns in data. Without it, the network would be limited in its ability to solve intricate problems, such as image recognition or natural language processing.

## Optimization in Neural Networks and Newton's Method

Newton's method is a powerful technique used for finding the zeros of a function, and we can also adapt it for optimization.

Imagine you're trying to find the lowest point in a hilly landscape. You start at a random spot (let's call it `X0`) and look around to see which direction goes downhill the fastest. You take a step in that direction to a new spot (`X1`), and then you repeat this process. Each time you take a step, you get closer to the lowest point. In Newton's method, we use the slope of the hill (the derivative) to help us find the next spot. The formula we use is:

`X(k+1) = X(k) - F(X(k)) / F'(X(k))`
Here, `F(X(k))` is the height of the hill at your current spot, and `F'(X(k))` is the slope. By using this formula, you can keep moving closer to the zero of the function, or in optimization, the minimum point of the function.

More formally,

- `X(k)` is the current approximation.
- `F(X(k))` is the value of the function at `X(k)`.
- `F'(X(k))` is the derivative of the function at `X(k)`.
- `X(k+1)` is the next approximation.

For optimization, when you want to minimize a function `G`, you would use the derivative of `G` (denoted as `G'`) in the formula:

`X(k+1) = X(k) - G'(X(k)) / (G'(X(k)))'`

Where:
- `X(k)` is the current approximation.
- `G'(X(k))` is the first derivative of `G` at `X(k)`.
- `G'(X(k))'` is the second derivative of `G` at `X(k)`.
- `X(k+1)` is the next approximation.

This iterative process helps you get closer to the zero of the function or the minimum point of the function.

### Second derivative
As we saw in newton's method, second derivative is very useful. This gives us a measure as to the amount the curve deviates from a straight line this is called the **`curvature`**

When you have positive second derivative its a convex function ,concave up or convex: function is increasing at an increasing rate i.e. `d^2(x)/dt^2 > 0` (local minimum)

We also have cocave down and that happens when the second derivative is negative `d^2(x)/dt^2 < 0` (local maximum)

`Leibniz notation d^2f(x)/dx^2 = d/dx(d(f(x))/ dx)`

`Lagrange notation: f''(x)`

We also come to know whether something is maximum or minimum for optimization type of problems. So just to recap between first and second derivative


| first derivative               | second derivative       |
|------------------------------- | ----------------------- |
| f'(0) > 0 increasing function  | f''(0) > 0 concave up, local minimum   |
| f'(0) < 0  decreasing function | f''(0) < 0 concave down, local maximum |

### Hessian
In this section we will be focussing on how secodn derivative is works for more than variables. Its actually a matrix full of second derivatives called the `Hessian`.

When we want to optimize a function with many variables, we can use multivariable Newton's method, and that one uses the Hessian.

|                                | Leibniz's notation        | Lagrange's notation      |
|------------------------------- | ----------------------- | -----------------------|
| Rate of change of f'x(x,y) wrt x  |  ∂^2f/∂x^2           | fxx(x,y)               |
| Rate of change of f'x(x,y) wrt y  |  ∂^2f/∂xy            | fxy(x,y)               |
| Rate of change of f'y(x,y) wrt x  |  ∂^2f/∂yx            | fyx(x,y)               |
| Rate of change of f'y(x,y) wrt y  |  ∂^2f/∂y^2           | fyy(x,y)               |

**`Hessian Matrix`**
Going from above, we can have a Hessian matrix as
```
H = [ 
        fxx(x, y)   fxy(x,y)
        fyx(x, y)   fyy(x,y)
    ]
```
So, our Hessian matrix will be the answer for a secodn derivative on a equation with say 2 variables, and to recap our first derivative for 2 variables will be 
```
[ 
    fx(x, y)
    fy(x, y)
]
```
### What does "concave up" means in 2 or more variables?

When the eigen value of a Hessian matrix are > 0 it means the matrix is positive definite, and the function is concave up, and 0.00 is minimum.

Similarly, when a matrix has all the eigen values negative, then its a concave down, and therefore 0.00 is maximum.

But, when we have a case when not all eigen values are either positive or not all negative there is neither a positive definite or negative definite. In this case a 0.0 is a saddle point, i.e. a point that is neither a minimum or maximum.

To conclude


|                       | 1 variable f(x)         | 2 variables f(x,y ) | More variables f(x1,x2,x...,xn) |
|-----------------------| ----------------------- | -----------------------------------------------|-----------------------|
| (local) minima        |  happy face f'(x) > 0   | upper paraboloid lambda1 > 0, lambd2 > 0       | All lambdai > 0 |
| (local) maxima        |  sad face f'(x) < 0     | down paraboloid lambda1 < 0, lambda2 < 0 | All lambdai < 0 |
| Need more info        |  f''(x)=0               | saddle point(lambda1 < 0 & lambda2 > 0 or lambda1 > 0 & lambda2 < 0) or some lambdai=0 | Some lambdai > 0 and some lambdaj < 0 or atleast one lambdai = 0 |

### Now, lets apply Newton's method to optimize functions with 2 or more variables

Recap our Newton's method for optimizing in 1 variable
`X(k+1) = X(k) - F'(X(k)) / F''(X(k))`
or rewriting by taking the inverse of the second derivative, and brining it to the numerator

`X(k+1) = X(k) - F''(X(k))^-1 F'(X(k)) `

To now determine the expression for Newton's method for 2 variables
the second derivative turns into a inverse Hessian matrix, and the first derivative as the gradient

```
[           [ 
  Xk+1    =     Xk     -  H^-1(Xk, Yk) F'(Xk, Yk)
  Yk+1          Yk  
]           ]    

```
**Note the order of the hessian matrix(2 by 2), and the matrix for the gradient(2 by 1)!**


### Here is a sample implementation of Newton's method on 1 variable, and 2 variables in comparison with Gradient descent
```
import numpy as np
'''
Let's optimize function  𝑓(𝑥)=𝑒^𝑥−log(𝑥)(defined for  𝑥>0) using Newton's method.
To implement it in the code, define function
𝑓(𝑥)=𝑒^𝑥−log(𝑥), & then its first and second derivatives
𝑓′(𝑥)=𝑒^𝑥 − (1/𝑥)
𝑓″(𝑥)=𝑒^𝑥 + (1/𝑥^2)
'''
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

def d2fdx2_example_1(x):
    return np.exp(x) + 1/(x**2)

x_0 = 1.6
print(f"f({x_0}) = {f_example_1(x_0)}")
print(f"f'({x_0}) = {dfdx_example_1(x_0)}")
print(f"f''({x_0}) = {d2fdx2_example_1(x_0)}")

def newtons_method(dfdx, d2fdx2, x, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - dfdx(x) / d2fdx2(x)
        print(x)
    return x

'''
In addition to the first and second derivatives, there are two other parameters in this implementation: number of iterations `num_iterations`, initial point `x`. To optimize the function, set up the parameters and call the defined function gradient_descent
'''

num_iterations_example_1 = 25; x_initial = 1.6
newtons_example_1 = newtons_method(dfdx_example_1, d2fdx2_example_1, x_initial, num_iterations_example_1)
print("Newton's method result: x_min =", newtons_example_1)

'''
Now, we will also implement gradient descent to compare how many iterations it takes. Note we know that gradient descent takes another param called "learning rate or alpha". We learn the disadvantages of gradient descent method in comparison with Newton's method: there is an extra parameter to control and it converges slower. However it has an advantage - in each step you do not need to calculate second derivative, which in more complicated cases is quite computationally expensive to find. So, one step of gradient descent method is easier to make than one step of Newton's method.
'''
def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
        print(x)
    return x

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
gd_example_1 = gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)
print("Gradient descent result: x_min =", gd_example_1) 

#Now lets redo for 2 variables
def f_example_2(x, y):
    return x**4 + 0.8*y**4 + 4*x**2 + 2*y**2 - x*y -0.2*x**2*y

def grad_f_example_2(x, y):
    return np.array([[4*x**3 + 8*x - y - 0.4*x*y],
                     [3.2*y**3 +4*y - x - 0.2*x**2]])

def hessian_f_example_2(x, y):
    hessian_f = np.array([[12*x**2 + 8 - 0.4*y, -1 - 0.4*x],
                         [-1 - 0.4*x, 9.6*y**2 + 4]])
    return hessian_f

x_0, y_0 = 4, 4
print(f"f{x_0, y_0} = {f_example_2(x_0, y_0)}")
print(f"grad f{x_0, y_0} = \n{grad_f_example_2(x_0, y_0)}")
print(f"H{x_0, y_0} = \n{hessian_f_example_2(x_0, y_0)}")

def newtons_method_2(f, grad_f, hessian_f, x_y, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - np.matmul(np.linalg.inv(hessian_f(x_y[0,0], x_y[1,0])), grad_f(x_y[0,0], x_y[1,0]))
        print(x_y.T)
    return x_y

num_iterations_example_2 = 25; x_y_initial = np.array([[4], [4]])
newtons_example_2 = newtons_method_2(f_example_2, grad_f_example_2, hessian_f_example_2, 
                                     x_y_initial, num_iterations=num_iterations_example_2)
print("Newton's method result: x_min, y_min =", newtons_example_2.T)

def gradient_descent_2(grad_f, x_y, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - learning_rate * grad_f(x_y[0,0], x_y[1,0])
        print(x_y.T)
    return x_y

num_iterations_2 = 300; learning_rate_2 = 0.02; x_y_initial = np.array([[4], [4]])
# num_iterations_2 = 300; learning_rate_2 = 0.03; x_y_initial = np.array([[4], [4]])
gd_example_2 = gradient_descent_2(grad_f_example_2, x_y_initial, learning_rate_2, num_iterations_2)
print("Gradient descent result: x_min, y_min =", gd_example_2) 
```


