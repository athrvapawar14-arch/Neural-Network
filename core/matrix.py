"""
First the constructor: it must take all 3 inputs to make is more easy to use. 

Second the internal representation : technically speaking, we should know what type of matrix we are dealing with. 
like a square one or something like that. so perhaps we would have to store that. 
also you said the matrix are like operators, and also called weights. perhaps we will need some other info. idk, this is still like maths for me. 
Ohhhh. i get it. the datatype. we have to ensure only the floats are allowed, so that info must also be stored. 
also idk if python allows Jagged matrices. so we will have to be careful with them. 

Third the property anotation : this will be needed to tell the size of the matrix. the rows x columns. we will return a tuple for this. 

Forth Operations : the basic add and substract, pretty standard. we will need a condition to check if the dimensions are safe first, so we will need property first. and then return a matrix
then the multiplication. first the scalar. we will need to handle the both M * 2 and 2 * M. and then return a matrix
and then the matrix multiplication. first check the condition for multiplication and then do it. and also we will need to configure the A x B and B x A restrictions. and then return a matrix 
though luckily in programming it may not be needed. 
the transpose will return a matrix which has rows and columns interchanged. 
then the determinant. if it is not zero then we can move with other operations. 
like the inverse of matrix. 
the same with subsequent features like iterable and indexable. they would return the element.
equality will most likely check each element one by one and return true or false boolean value.
copy. we will need to see if they need a shallow copy or a deep copy. based on it we will either return a new reference variable or new matrix.
identity will need the matrix to be square, hence that will need some work. 
same with zero. though it would need some work. and then ones. 


also the whole thing class must be able to handle vector inputs. cause technically vectors are one row or one column matrices. 

Big Design Question : it should return a vector. i am stating this cause vector will be light weight compared to matrix objects. 
and also i just said vectors can be considered as matrices as well. it will be smart usage of them. though maybe making vector class a child of matrix class is also an option. 



"""

class Matrix:
    pass