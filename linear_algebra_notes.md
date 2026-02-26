
Linear Algebra
Vectors
- Vectors are usually viewed by computers as an ordered list of numbers which they can perform "operations" on - some operations are very natural
- Something which moves in a space of fitting parameters.
- vector addition is associative, (r+s)+t == r+(s+t)
- multiplication by a scalar so when we multiply by a scalar we multiply all the components of a vactor with the scalar, so if r=2i+3j, then 2r = 4i+6j
- substraction of vectors, say s=-1i+2j, r=3i+2j, then r-s = r+(-1*s) = 3-(-1)i+(2-2)j = 4i+0j
Modulus and dot product of vectors:
Size of vector = Sum of its squares of its components.
For ex. in a 2D for a vector r(ri+bj), the size of this vector will be=sqrt(ri*ri+bj*bj)

Dot product of 2 vectors r, and s= ri*si + rj*sj
        - properties of dot product:
            - commutative: r.s == s.r
            - distributive over addition: r.(s+t) = r.s + r.t
            - associative over scalar multiplication: r.(as) = a(r.s)
            - r.r = r1.r1+r2.r2+...rn.rn = sqrt(r^2+r^2)^2 = |r^2|
Cosine and Dot products
        - cosine rule:
cosine rule for 3 sides of a triangle a,b,c where theta is the angle between a and b c^2 = a^2+b^2-2*a*b*cos(theta)
            - The cosine rule provides a direct relationship between the lengths of the sides of a triangle and the cosine of one of its angles. This is crucial when working with vectors, as it helps to relate geometric properties to algebraic expressions.
            - By using the cosine rule, we can express the dot product of two vectors in terms of their magnitudes and the angle between them. This is foundational in understanding how vectors interact in space.The cosine of the angle between two vectors gives you valuable insight into their interaction, such as whether they are pointing in the same direction, opposite directions, or are orthogonal (perpendicular) to each other.
            - r.s = |r||s|* cos(theta), so when angle is 90, they go in 90 deg, when angle is 180, they are opposite, and when angle=0 they both go in same direction
         - Another way to write the above equation, is cos(theta) = r.s/|r||s|
Projection
        - The cosine rule when applied gives us a projection of s on r, to mention it in a different way it gives us the "shadow of s on r" or "how much of s is onto r"
        - however is angle between s and r is 90, then there won't be any "shadow" of s on r
Scalar Projection
scalar projection when we divide the dot product with |r| becomes r.s / |r| = |s|.cos(theta), so in the case when angle between s and r is 90, then scalar projection of s onto r is 0
Vector projection
            - The vector projection of vector S onto vector R gives you a new vector that points in the direction of R and has a length that represents how much of S goes in the direction of R.
To find the projection of vector s onto r we use: s.r/|r|^2 * r

Changing the reference frame

Changing basis
The focus is on understanding the coordinate system in vector spaces, how to project vectors, and the significance of basis vectors in describing vectors in different coordinate systems.
* Understanding Coordinate Systems
    - A vector can be represented in different coordinate systems, defined by basis vectors, which can be arbitrary and not necessarily orthogonal.
    - The representation of a vector, such as r, can change based on the choice of basis vectors, like e1 and e2 or b1 and b2.
* Projection and Basis Vectors
    - When basis vectors are orthogonal, the dot product can be used to project vectors efficiently, simplifying calculations.
   - The projection of a vector onto a basis vector gives a scalar projection and a vector projection, which can be summed to find the vector in the new basis.
* Verifying Orthogonality
    * To check if two basis vectors are orthogonal, the dot product can be calculated; if the result is zero, the vectors are at 90 degrees to each other.
    * This orthogonality allows for straightforward transformations between different basis representations of the same vector.

Significance of Basis vectors in vector representation
Basis vectors are crucial in vector representation for several reasons:
* Defining Coordinate Systems: Basis vectors establish the framework for a coordinate system in which vectors can be expressed. They determine how we describe the position and direction of vectors in space.
* Flexibility in Representation: A vector can be represented in multiple ways depending on the choice of basis vectors. This flexibility allows for different perspectives and simplifications in calculations.
* Orthogonality and Simplification: When basis vectors are orthogonal (at 90 degrees to each other), calculations involving projections and transformations become simpler and more efficient. This is particularly useful in machine learning and data analysis.
* Dimensionality: The number of basis vectors defines the dimensionality of the vector space. For example, in a 3D space, three basis vectors are needed to represent any vector within that space.
* Transformation Between Spaces: Basis vectors allow for the transformation of vectors from one coordinate system to another. This is essential in various applications, such as computer graphics, physics, and machine learning.

Significance of orthogonality in basis vectors for calculations
Orthogonality in basis vectors is significant for calculations for several reasons:
* Simplified Calculations: When basis vectors are orthogonal (at 90 degrees to each other), the dot product between them is zero. This property simplifies many calculations, especially when projecting vectors onto these basis vectors.
* Independent Contributions: Orthogonal basis vectors ensure that each vector contributes independently to the representation of another vector. This means that the influence of one basis vector does not affect the others, making calculations more straightforward.
* Efficient Projections: In cases where basis vectors are orthogonal, the projection of a vector onto a basis vector can be computed easily using the dot product. This leads to faster and more efficient computations, which is particularly beneficial in machine learning algorithms.
* Numerical Stability: Orthogonal basis vectors can help reduce numerical errors in calculations, especially in high-dimensional spaces. This stability is crucial when dealing with large datasets or complex mathematical operations.
* Geometric Interpretation: Orthogonality provides a clear geometric interpretation of vector relationships, making it easier to visualize and understand the structure of the vector space.

Relationship between basis vectors and dimensionality of a vector space
The relationship between basis vectors and the dimensionality of a vector space is fundamental and can be summarized as follows:
* Definition of Dimensionality: The dimensionality of a vector space refers to the number of independent directions or dimensions in that space. It is defined by the number of basis vectors required to span the entire space.
* Basis Vectors and Span: A set of basis vectors is a minimal set of vectors that can be combined (through linear combinations) to represent any vector in the space. The number of basis vectors in this set directly corresponds to the dimensionality of the vector space.
* Example:
    * In a 1D space, you need 1 basis vector (e.g., a line).
    * In a 2D space, you need 2 basis vectors (e.g., a plane).
    * In a 3D space, you need 3 basis vectors (e.g., the three-dimensional space we live in).
* Independence: The basis vectors must be linearly independent, meaning no vector in the set can be expressed as a linear combination of the others. This independence ensures that each basis vector contributes a unique direction to the space.
* Higher Dimensions: In higher-dimensional spaces, the same principle applies. For example, in a 4D space, you would need 4 linearly independent basis vectors.

Linear independence
* refers to a set of vectors being independent if no vector in the set can be expressed as a linear combination of the others. This means that each vector adds a new dimension to the space they occupy.
* When the vectors (or equations) are linearly independent, it often indicates that the system has a unique solution. Conversely, if the vectors are dependent, it can lead to either no solutions or infinitely many solutions.

Singularity of matrix
Determining Singularity of a Matrix
A matrix is considered singular if it does not have an inverse. This typically occurs when:
* The determinant of the matrix is zero.
* The rows (or columns) of the matrix are linearly dependent, meaning one row (or column) can be expressed as a linear combination of others.
Remember: When the determinant of a matrix is zero, it indicates that the matrix is singular, meaning it does not have an inverse. 
This typically implies that the system of equations represented by that matrix either has no solutions or infinitely many solutions.
To check for singularity:
* Calculate the determinant. If it's zero, the matrix is singular.
* Alternatively, you can perform row reduction to see if any row becomes a zero row, indicating linear dependence.

System of linear equations
Differences Between a Linear System of Equations having Infinitely many Solutions and No Solutions:
* Infinite Solutions:
    * Occurs when the equations are dependent, meaning they represent the same line or plane in space.
    * Example: The equations (2x + 3y = 6) and (4x + 6y = 12) are dependent. They represent the same line, leading to infinitely many solutions.
* No Solutions:
    * Occurs when the equations are inconsistent, meaning they represent parallel lines or planes that never intersect.
    * Example: The equations (2x + 3y = 6) and (2x + 3y = 12) are inconsistent. They represent parallel lines, leading to no solutions.
* So, no solution ==> parallel, infinitely many solutions ==> a single line, a single solution ==> a point of intersection

Solving System of linear equations
* Row Echelon Form: A matrix is in this form when it has ones on the main diagonal and zeros below it. This form provides useful information about the system of equations.
* Reduced Row Echelon Form: This is a further simplification where the matrix has ones on the diagonal and zeros everywhere else.
Row operations that preserve singularity of a matrix
* row switching preserves the singularity/non-singularity of a matrix
* multiplying a row by scalar(non-zero) even then it turns a non zer determinant to a non-zero determinant, and a zero determinant to a zero determinant
* adding one row to other we get same determinant
Remember: However, multiplying a row by zero (option 4) would make the matrix singular, as it would create a row of zeros, indicating linear dependence.
Rank of a matrix
* amount of information a system carries is the "rank" of a system, i.e. The rank is the largest number of linearly independent rows/columns in the matrix.
* there exists a relationship between rank of a matrix, and solutions to the system
    * A matrix is singular IFF its rank == number of rows i.e. it carries maximum amount of information as possible
* A rank of 0 means that a matrix or system of equations carries no information at all. This situation occurs when all the rows (or equations) in the matrix are either zero or redundant, meaning they do not provide any unique information.
*  the rank of a matrix relates to the linear independence of its rows and concluded that dependent rows indicate a singular matrix.

Row Echelon form of a matrix
* The row echelon form can have different representations depending on whether you normalize the rows to have leading coefficients of 1 or not.
* The important part is that the structure of having zeros below the leading coefficients is maintained.
	The rank of the matrix is the number of non-zero rows in the row echelon form. To find the rank we need to perform the following steps:
    * Find the row-echelon form of the given matrix
    * Count the number of non-zero rows.
* A matrix is in Row Echelon form if it has the following properties:
    * Zero Rows at the Bottom: If there are any rows that are completely filled with zeros they should be at the bottom of the matrix.
    * Leading 1s: In each non-zero row, the first non-zero entry (called a leading entry) can be any non-zero number. It does not have to be 1.
    * Staggered Leading 1s: The leading entry in any row must be to the right of the leading entry in the row above it.

Reduced Row Echelon form
- the matrix is already in the row echelon form
- each pivot is a 1
- any number above a pivot is a 0
- rank of the matrix is equal to the number of pivots

Gaussian elimination method
* Introduction of the augmented matrix, which includes both the coefficients of the variables and the constants from the right side of the equations. This allows us to work with the entire system in a single matrix format.
* By applying row operations to the augmented matrix, we can manipulate the entire system of equations simultaneously, making it easier to find solutions.
* After transforming the matrix into reduced row echelon form, we can use back substitution to find the values of the variables, taking into account the constants.
* If the system is singular, then there will be a row containing all 0s so we can stop there.So, to summarize:
    * row full of zeros -> constant in that row is 0 -> infinitely many solution(think: 0a+0b+0c=0)
    * row full of zeros in row echelon form -> constant in that row is non-zero -> no solution(eg: 0a+0b+0c = 42)

Vectors
L1 norm
    - sum of absolute value of the components of a given vector, so say u = (1,5), and v=(6,2) L1 = abs(6-1, 2-5) = abs(5) + abs(-3) = 8 
L2 norm
    - square root of the sum of squares of all the components of the vector, say u = (1,5), and v = (6, 2) L2 = sqrt((6-1)^2, (2-5) ^ 2) = sqrt(25+9) = sqrt(34) = 5.83
    - Or  for a vector v=(x,y,z) => ||v|| = sqrt(x^2 + y^2 + z^2)
- multiplying a vector by scalar(positive or negative):u=(1,2), scalar=-2, then output=(-2,-4)
Dot product
    - L2 norm is the dot product of the vector with itself
    - Dot product of 2 vectors say a and b is: a.b = (ax * bx) + (ay * by)+(az * bz)
    - also denoted as <x, y>
- Geometric dot product
    - Orthogonal vectors have a dot product of 0, i.e. if the angle between the 2 vectors is 90 deg.
- If u and v have an angle theta between them, then the dot product of u and v is <u,v> = <u', v> = |u|.|v|.cos(theta)
- Multiplying a matrix with a vector
- Dot product of 2 matrices say 
w=[-9 
      -1]
and
v = [-3
	-5]
then, w . v = (-9)*(-3) + (-1 * -5) = 27 + 5 = 32
Distance between 2 vectors
-  u, and v is given by d(u, v) = sqrt((ux1 - vx1)^2 + (ux2 - vx2)^2 +.. (uxn - vxn)^2)

Magnitude of a vector from vector A to vector B
- is given as : sqrt((Ax1 - Bx1)^2 + (Ax2 - Bx2)^2 +.. (Axn - Bxn)^2)

Matrices as Linear Transformations
- matrix times the point will give us the new coordinates for the same point in the linear transformation

Linear Transformations as Matrices
- You basically start with a linear transformation, and then you try to find the matrix
- you look at where the two fundamental vectors (1,0), and (0, 1) go, and those are your columns  of the matrix
- multiplying 2 matrices A(m*n), and B(n*p) only possible if no.oif columns in matrix A == no.of rows in matrix B, and the resultant matrix has dimension of m*p
- In case of an IDENTITY matrix, it has all 1s on its diagonals, and zeros everywhere else denoted as I. So, A * I = A
- A*A^-1 = I, so to calculate the matrix inverse of a 2*2 matrix we have a 2*2 matrix 
		A=[a11   a12
              a21   a22]
Then determinant Det = 1 / a11 - a22 -- step(1)
Then, from step1 above
if and only if Det != 0 then
A^-1 = Det * [a22   -a12
			  -a21   a11]
- Which matrices have an inverse?
    - Non-singular matrices always have an INVERSE, these are also called INVERTIBLE matrix
    - Also, for INVERTIBLE matrices the DETERMINANT is NON-ZERO
- A transformation T is said to be said to be linear if the following two properties are true for any scalar  𝑘, and any input vectors 𝑢 and v:
    - T(kv) = kT(v)
    - T(u+v) = T(u) + T(v)
More notes on determinant and its relation with transformation
* Area Scaling Factor: The absolute value of the determinant (in this case, 2) represents the scaling factor of the area when the transformation is applied to a shape (like a parallelogram) in the plane. A determinant of (-2) means that the area is scaled by a factor of 2, and the negative sign indicates that the transformation also involves a reflection across a line.
* Invertibility: Since the determinant is non-zero, this matrix represents an invertible transformation. This means that there is a unique output for every input, and you can reverse the transformation.
* 

Principal Component Analysis
- reduce dimension of a dataset while preserving as much information as possible

Basis vectors
The vectors (1,0) and (0,1) are known as the basis vectors in a two-dimensional space. Here’s why they are particularly useful:
* Simplicity: These vectors are simple and easy to work with. They represent the x-axis and y-axis in a Cartesian coordinate system, making calculations straightforward.
* Linear Independence: The vectors (1,0) and (0,1) are linearly independent, meaning that no vector can be expressed as a linear combination of the other. This property is essential for defining a basis.
* Span the Space: Together, these vectors can be used to represent any vector in the two-dimensional space. Any vector (x,y) can be expressed as a combination of (1,0) and (0,1) as follows: [ (x,y) = x \cdot (1,0) + y \cdot (0,1) ]
Using these basis vectors allows us to analyze transformations and areas in a clear and consistent manner.
Basis vectors play a crucial role in linear algebra for several reasons:
* Defining Vector Spaces: Basis vectors define a vector space. A set of basis vectors can be used to represent any vector in that space through linear combinations.
* Dimensionality: The number of basis vectors in a set corresponds to the dimension of the vector space. For example, in a 2D space, two basis vectors are needed, while in a 3D space, three are required.
* Linear Independence: Basis vectors are linearly independent, meaning no vector in the set can be expressed as a combination of the others. This property ensures that each vector contributes uniquely to the space.
* Simplifying Calculations: Using basis vectors simplifies calculations in linear transformations, making it easier to understand how vectors change under various operations.
* Coordinate Systems: Basis vectors provide a framework for defining coordinates in a space, allowing for the representation of vectors in a standardized way.
Basis vectors are fundamental to understanding linear transformations in linear algebra. they relate:
* Transformation of Basis Vectors: A linear transformation can be thought of as a function that takes vectors from one space and maps them to another. When you apply a linear transformation to the basis vectors, the resulting vectors define how the entire space is transformed. For example, if you have a transformation matrix ( A ) and apply it to the basis vectors (1,0) and (0,1), the transformed vectors ( A(1,0) ) and ( A(0,1) ) will determine the new orientation and shape of the space.
* Representation of Any Vector: Any vector in the space can be expressed as a linear combination of the basis vectors. When a linear transformation is applied, the transformation of the entire vector can be understood by looking at how the basis vectors are transformed. This means that knowing how the basis vectors change under the transformation allows you to predict how any vector will change.
* Matrix Representation: Linear transformations can be represented using matrices. The columns of the transformation matrix correspond to the images of the basis vectors under the transformation. This matrix representation simplifies calculations and helps visualize the transformation.
* Preservation of Structure: Linear transformations preserve the linear structure of the space. This means that operations like addition and scalar multiplication behave consistently when applied to vectors, which is crucial for maintaining the properties of the vector space.

Determinant of product of matrices
- Det(A) . Det(B) = Det(AB) i.e. determinant of product of matrices(Dc) == product of determinants of the matrices(Da, and Db). Also, if either of A or B is a singular matrix, then Det(AB)=0 so AB is a singular matrix
- Det(A^-1) = 1/Det(A)
- But why is that?
    - we know Det(AB) = Det(A).Det(B)
    - say B = A^-1 then,
    - Det(A.A^-1) = Det(A).Det(A^-1),
    - we know A.A^-1 = I(the identity matrix), so substituting
    - Det(I) = Det(A).Det(A^-1), we know Det(I) == 1, so substituting
    - 1 = Det(A).Det(A^-1) we get
    - Det(A^-1) = 1/Det(A)
Understanding the concept of determinants in linear transformations is important for several reasons:
* Geometric Interpretation: Determinants provide insight into how transformations affect areas and volumes. For example, a determinant greater than one indicates that the transformation enlarges areas, while a determinant less than one indicates a reduction.
* Matrix Properties: Knowing the determinant helps in determining whether a matrix is singular (non-invertible) or non-singular (invertible). This is crucial in solving systems of linear equations and understanding the behavior of linear transformations.

Span in Linear Algebra
- A span of a set of vectors is simply the set of points that can be reached by walking in the direction of these vectors
- A basis is a minimal spanning set, contains linearly independent set of vectors
- length of a basis(i.e. the number of vectors in the basis) is equal to the dimensions of the space it spans
    * Dimension of a space: This refers to the number of independent directions in that space. For example, a line has a dimension of 1, a plane has a dimension of 2, and a three-dimensional space has a dimension of 3.
    * Length of a basis: This is the number of vectors in a basis set. A basis must span the space and consist of linearly independent vectors.
* Not all sets of N vectors are a basis for N-dimensional space
* Basis Vectors and Span: A set of basis vectors is a minimal set of vectors that can be combined (through linear combinations) to represent any vector in the space. The number of basis vectors in this set directly corresponds to the dimensionality of the vector space.
* Example:
    * In a 1D space, you need 1 basis vector (e.g., a line).
    * In a 2D space, you need 2 basis vectors (e.g., a plane).
    * In a 3D space, you need 3 basis vectors (e.g., the three-dimensional space we live in).
* Independence: The basis vectors must be linearly independent, meaning no vector in the set can be expressed as a linear combination of the others. This independence ensures that each basis vector contributes a unique direction to the space.
* Higher Dimensions: In higher-dimensional spaces, the same principle applies. For example, in a 4D space, you would need 4 linearly independent basis vectors.
* More on this:
    * If three vectors span a plane, it indicates that they are linearly dependent. Here’s what that means:
    * Linear Dependence: At least one of the three vectors can be expressed as a linear combination of the other two. This means that they do not add any new direction to the space they occupy.
    * Dimensionality: A plane is a two-dimensional space, so having three vectors that span it suggests that they do not extend beyond that plane. Essentially, you can represent any point in the plane using just two of those vectors.

Eigen Values and Eigen Vectors
- Eigenvalues and Eigenvectors help us understand how data is spread out in different directions. Imagine you have a bunch of balloons tied together in a bunch. When you pull on one side, the balloons stretch in that direction. The direction in which they stretch the most is like the Eigenvector, and how much they stretch is like the Eigenvalue. 
- So, Eigenvectors tell us the directions of the most significant stretches in our data, while Eigenvalues tell us how much stretching happens in those directions.
- To visualize this, think of a circle that represents your data. When you apply a transformation (like pulling on the balloons), that circle can turn into an ellipse. The longest part of the ellipse shows the direction where the data is most spread out, which corresponds to the Eigenvector with the largest Eigenvalue. This helps us identify the most important features in our data, making it easier to analyze and understand.
- A*v = lambda*v for each eigenvector/eigenvalue
- Eigen vectors= direction of stretch
- Eigen values = how much stretch
- Finding Eigen values & eigen vectors
    * Consider matrix A = [ [4 1 ], [2, 3]]
        * Step 1: Calculate Eigenvalues
        * To find the eigenvalues, you need to solve the characteristic equation:
        * Subtract λ (lambda) from the diagonal elements of the matrix A:
            * A = [[4 - lambda, 1], [2, 3 - lambda]]
        * Step 2: 	Calculate the determinant of the resulting matrix:
            * (4-λ)(3-λ) - (2*1) = 0, solve for lambda to get values 5 and 2 respectively.
Find Eigenvectors
    * Now, find the eigenvectors for each eigenvalue.
    * For λ1 = 5:
        * Substitute λ1 into the matrix, to get A = [[4 - 5, 1], [2, 3 - 5]] => [[-1, 1], [2, -2]], and solve the system of equations as -1x + 1y = 0 and 2x - 2y = 0 we get x = y The eigenvector corresponding to λ1 = 5 is any scalar multiple of (1, 1).
    * For λ1 = 2
        * Substitute λ2 into the matrix, to get A = [[4 - 2, 1], [2, 3 - 2]] => [[2, 1], [2, 1]], and solve the system of equations as 2x + 1y = 0 and 2x + 1y = 0 we get x =-0.5y The eigenvector corresponding to λ1 = 2 is any scalar multiple of (-0.5 1).
* For a 2x2 matrix, different eigenvalues yield distinct eigenvectors.
* For a 3x3 matrix, the number of eigenvectors can vary based on the eigenvalues' uniqueness or repetition.

Dimensionality Reduction and Projection
Dimensionality reduction
- reduce number of cols while preserving as much information as possible
- fewer features, but same number of observations
- leads to smaller datasets, easier to visualize, and easier to manage
Projections
- To project a matrix A onto direction given by the vector v, you first need to multiply matrix A by vector v. However, first you need to scale vector v so that it has norm 1, so divide v by its own L2 norm ||v| so, Ap = A * v / ||v||^2
- So,  Ap = A* v, and to project matrix A onto vectors v1, and v2, Ap = A(v1/||v1|| v2/||v2||)

Variance
- Variance measures how much the values in a dataset differ from the mean (average) of that dataset.
- It gives you an idea of the spread or dispersion of the data points.
- Average squared distance from the mean, so if the average is farther from the mean, variance will increase
- Variance focuses on a single variable's spread.
- A variance of zero means there is no spread or dispersion in the data. All data points are identical.
- Steps:
    - find the mean(u) = (X1 + X2 +.....+XN) / N
    - Subtract the mean from each value to calculate the difference:
        - X1-u, X2-u, ...XN-u
    - Square each of the differences calculated in previous step:
        - (X1-u)^2, (X2-u)^2, ..., (XN-u)^2
    - Find the average of the squared differences
        - Add all the squared differences together
        - Divide by the no.of values N for population variance, or by N - 1 for sample variance
            - [(X1 - u)^2 + (X2 - u)^2 + (X3- u)^2 + ....+(XN - u)^2] / N
- 1/n-1 * sum(1 to n)[xi - mean(x)]^2

Covariance
- measures how 2 features of a dataset varies wrt one another or the direction of the relationship between 2 variables
- Eg: Imagine you have a group of friends, and you want to see how their heights and weights are connected. The covariance matrix helps you see if taller friends tend to weigh more, weigh less, or if there's no clear pattern at all.
- negative covariance ==> negative trend
- small value of covariance ==> denotes flat trend or no relationship
- positive value indicates indicates positive trend 

Covariance Martrix
- To create a covariance matrix, you first calculate the variance for each variable (like height and weight) and the covariance between each pair of variables. The variance tells you how spread out the values are, while covariance shows the direction of the relationship. 
- If the covariance is positive, it means that as one variable increases, the other does too. 
- If it's negative, it means that as one variable increases, the other decreases.
- Finally, you organize these values into a square matrix, where the diagonal contains the variances and the off-diagonal contains the covariances.
- calculate Var(x), Var(y), and covariance(x,y)
- note Cov(x, y) == Cov(y, x)
- and Cov(x, x) == Var(x)
C = [Cov(x, x)       Cov(x, y) ]
         Cov(y, x)       Cov(y, y) ]
- Matrix formula
Say A - u(read as mu) = [x1-ux   y1-uy
					         .....        ......
					        xn-ux    yn-uy]
C = 1/n - 1 (A-u)^T(A-u)
Steps:
- arrange the data with a different feature in each column
- Calculate the column averages(u)
- Substract each average from their respective column to getneare A - u
- Calculate Covariance matrix using the above formula

Principal Component Analysis(PCA)
- simplify complex datasets while preserving as much information as possible
- goal of the PCA is to find the projection that preserves the max possible spread in your data, even as you reduce the  dimensionality of your data set.
* Dimensionality Reduction:
    * PCA reduces the number of variables (dimensions) in a dataset while retaining the essential information. This makes it easier to visualize and analyze data.
* Preserving Variance:
    * It identifies the directions (principal components) in which the data varies the most. By projecting the data onto these directions, PCA helps maintain the maximum variance.
* Noise Reduction:
    * By focusing on the most significant components, PCA can help filter out noise and less important features, leading to cleaner data for analysis.
* Visualization:
    * PCA allows for the visualization of high-dimensional data in lower dimensions (like 2D or 3D), making it easier to identify patterns, clusters, or trends.
* Feature Extraction:
    * It transforms the original features into a new set of features (principal components) that can be more informative for machine learning models.
* Improving Model Performance:
    * By reducing dimensionality and focusing on the most important features, PCA can enhance the performance of machine learning algorithms by reducing overfitting and improving training speed

PCA Mathematical formulation
    - 1. create matrix (X) of your data 1 for each variable/feature, and n rows
    - 2. center your data
        - calculate column averages(u), and subtract them from each col giving you the matrix X - u
    - 3. Calculate the covariance matrix using the formula: C = 1 / n - 1 (X - u)^T (X - u)
    - 4. Calculate Eigen vectors and eigen values for the covariance matrix
    - 5. sort the eigenvalues in DESC order
    - 6. select the top K eigen vectors
    - 7. create projection matrix which has 2 columns where each is one of the eigenvector you choose, scaled by its norm V = [ v1/||v1||, v2 / ||v2|| ]
        * This projection matrix (V) is crucial as it defines how the original data is transformed into the new lower-dimensional space.
    - 8. project centered data
        - XPCA = (X-u)*V

Discrete Dynamical Systems
A discrete dynamical system describes a system where, as time goes by, the state changes according to some process. When defining this dynamical systems you could represent all the possible states, such as sunny, rainy or cloudy, in a vector called the state vector.
Each discrete dynamical system can be represented by a transition matrix 
𝑃 which indicates, given a particular state, what are the chances or probabilities of moving to each of the other states. This means the element (2,1) of the matrix represents the probability of transitioning from state 1 to state 
2.
Starting with an initial state  𝑋0, the transition to the next state 𝑋1 is a linear transformation defined by the transition matrix 
𝑃: 𝑋1 =𝑃𝑋0 . That leads to 𝑋2=𝑃𝑋1=𝑃^2𝑋0  𝑋3=𝑃3𝑋0, and so on. This implies that 𝑋𝑡=𝑃𝑋𝑡−1 for 𝑡=0,1,2,3,…
In other words, we can keep multiplying by P to move from one state to the next.
One application of discrete dynamical systems is to model browsing web pages. Web pages often contain links to other pages, so the dynamical system would model how a user goes from one page to another by hopping from link to link.



#linalg
