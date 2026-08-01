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

from core.vector import Vector
import numpy as np

class Matrix:   

    def __init__(self, *args):

        processed_args = []

        # Convert Vector instances to their inner numpy arrays
        for arg in args:
            if isinstance(arg, Vector):
                processed_args.append(arg._arr)
            else:
                processed_args.append(arg)

        if len(processed_args) == 1:
           data = processed_args[0]
            # single argument: could be nested list, or a flat list (vector)
           data_arr = np.array(data, dtype=float)

           if data_arr.ndim == 1:
                data_arr = np.atleast_2d(data_arr)
           self._data = data_arr
        
        else:
            # multiple arguments: each one is a row / this may also handle vector inputs as vector is iterable. 
            self._data = np.array(processed_args, dtype=float)
        

    # ----- Properties-------

    @property
    def shape(self):
        return self._data.shape

    @property
    def data(self):
        """Read-only access to the underlying numpy array."""
        return self._data

    @property
    def T(self):
        return Matrix(np.transpose(self._data))

    @property
    def det(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Determinant is only defined for square matrices.")

        return np.linalg.det(self._data)

    @property
    def I(self):

        # first check for squareness
        if self.shape[0] != self.shape[1]:
            raise ValueError("Inverse is only defined for square matrices.")

        # now check for singulatity. 
        if np.isclose(self.det, 0):
            raise ValueError("Cannot invert singular matrix (determinant is zero).")
    
        return Matrix(np.linalg.inv(self._data))

    # ----- Operations --------

    def __repr__(self):
        formatted_ele = ",\n ".join(str(x) for x in self._data)
        return f"Matrix({formatted_ele})"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        if isinstance(other, Matrix):
            return np.array_equal(other._data, self._data)
            

    def __add__(self, other):
        if isinstance(other, Matrix) and self.shape == other.shape:
            return Matrix(self._data + other._data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other,Matrix) and self.shape == other.shape:
            return Matrix(self._data - other._data)
        return NotImplemented

    
    def __mul__(self, other):
        """Scalar multiplication: v * 3"""
        if isinstance(other, (int, float, np.number)):
            return Matrix(self._data * other)
        return NotImplemented
    
    def __rmul__(self, other):
            """Reverse scalar multiplication: 3 * v"""
            return self.__mul__(other)

    def __matmul__(self, other):
        # for matrix multiplication.
        if isinstance(other,Matrix) and self.shape[1] == other.shape[0]:
            return Matrix(np.matmul(self._data, other._data)) 

        # for vector multiplication.
        if isinstance(other, Vector) and self.shape[1] == len(other):
            return Vector(self._data @ other._arr)