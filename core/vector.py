"""

constructor : 
i want the users to directly use it, like Vector(1,2,3)
so i will keep default values at creation as 0. so in case someone just uses one variable, 
they can do other operations. 
and during construction itself they values will be stored. 
perhaps i will make it variable argument function, as parameters can be a lot more than just 3. 

print:
the print function must display the list of all the different components of the vector. like i know what vectors are 
i studied about them in maths and physics. so i know that we are trying to represent them in a program. 
vectors have magnitude and directions. and of course the unit vectors of each direction. 
i wanted to suggest printing as like 1i + 2j + 3k notation. but it makes no sense in coding. and we may have variable parameters as i mentioned. 

Operations: 
this is were we use operator overloading. here i would have to ensure the objects act the way i want when given operators are used. 
so standard vector operations are 
addition, substraction, there is cross and dot products. there is magnitude. and then there is unit vector. idk about others.

Indexing:
techincally this should work. cause we are storing the inputs in a list. so it is indexable. 

Iteration:
even interation should work. 

Validation:
yes. as i mentioned, techincally we can add vectors of different sizes/directions. 


"""

import numpy as np

class Vector:
    def __init__(self, *args):
        # Allow Vector(1, 2, 3) OR Vector([1, 2, 3])
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            self._arr = np.array(args[0], dtype=float)
        else:
            self._arr = np.array(args, dtype=float)

    # --- Properties ---
    @property
    def magnitude(self):
        """Calculates vector magnitude (length)."""
        return np.linalg.norm(self._arr)

    @property
    def unit(self):
        """Returns normalized unit vector."""
        mag = self.magnitude
        if mag == 0:
            raise ValueError("Cannot calculate the unit vector of a zero vector.")
        return Vector(self._arr / mag)

    # --- Operator Overloading ---
    def __add__(self, other):
        if isinstance(other, Vector) and len(self) == len(other):
            return Vector(self._arr + other._arr)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector) and len(self) == len(other):
            return Vector(self._arr - other._arr)
        return NotImplemented

    def __matmul__(self, other):
        """Dot product: v1 @ v2"""
        if isinstance(other, Vector) and len(self) == len(other):
            return np.dot(self._arr, other._arr)
        return NotImplemented

    def __mul__(self, other):
        """Scalar multiplication: v * 3"""
        if isinstance(other, (int, float, np.number)):
            return Vector(self._arr * other)
        return NotImplemented

    def __rmul__(self, other):
        """Reverse scalar multiplication: 3 * v"""
        return self.__mul__(other)

    def cross(self, other):
        """3D Vector Cross Product"""
        if isinstance(other, Vector) and len(self) == 3 and len(other) == 3:
            return Vector(np.cross(self._arr, other._arr))
        raise ValueError("Cross product requires two 3D vectors.")

    # --- Indexing & Representation ---
    def __getitem__(self, key):
        return self._arr[key]

    def __setitem__(self, key, value):
        self._arr[key] = value

    def __repr__(self):
        # Clean formatting like Vector(1.0, 2.0, 3.0)
        formatted_elems = ", ".join(str(x) for x in self._arr)
        return f"Vector({formatted_elems})"

    def __iter__(self):
        return iter(self._arr)

    def __len__(self):
        return len(self._arr)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return np.array_equal(self._arr, other._arr)
        return False    