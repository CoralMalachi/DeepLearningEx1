import numpy as np

STUDENT={'name': 'Coral Malachi_Daniel Braunstein',
         'ID': '314882853_312510167'}

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    # readwrite flag - By default, the nditer treats the input array
    # as a read-only object. To modify the array elements, you must specify 
    #either read-write or write-only mode.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index #index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        ### YOUR CODE HERE:
        
        #create copy of x vector
        x_minus_epsilon = x.copy()
        x_plus_epsilon = x.copy()

        x_minus_epsilon[ix] = x_minus_epsilon[ix] - h
        x_plus_epsilon[ix] = x_plus_epsilon[ix] + h

        #computing derivative by the formula according Gradient checking
        #[0] gives us the value
        numeric_gradient = f(x_plus_epsilon)[0] - f(x_minus_epsilon)[0]
        #print(f(x_plus_epsilon)[0])
        numeric_gradient = numeric_gradient/(2*h)

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient)
            return
    
        it.iternext() # Step to next index

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    print ""

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
