# %%
""" My first cat image classifier
    From Andrew ng's neural network deep learning specialization
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy;
from PIL import Image
from scipy import ndimage
import random 

def print_image(index):
    index = 25
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    plt.imshow(train_set_x_orig[5])
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")




def sigmoid(z):
    """ convert a number to sigmoid
        input: int or float
        returns: sigmoid figure which is a float
    """
    return 1/(1+np.exp(-z))


def zero_param_init(dim):
    """ initialize parameters to zero
        input: int indicating cprresponding to the number of zeros to create
        returns: initialized weight vector (w), and initialized bias (b )
    """
    b = 0
    w = np.zeros((dim,1))
    return w, b

def random_param_init(dim):
    """ randomly initialize parameters to values close  to zero
        input: int indicating corresponding to the number of random figures to create
        returns: initialized weight vector (w), and initialized bias (b)
    """
    # TODO
    pass

def flatten_and_standardize(data):
    """ Flattens data so that dimensions would be compatible during vectorized operations"""
    return data.reshape(data.shape[0], -1).T/255
# def standardize(data):
#     """ Assumes that data is a numpy array
#     """
#     return data/255

def propagate(w, b, X, Y):
    """ This carries out forward, and back propagation, in the network
        All operations within this function are vectorized implementations
        input: weight parameters and bias(w, b).
               input features and, output values(X, y).
               All input values except b are vectors
        returns: cost function and dictionary containing differential vectors of w amd b
    """
    m = X.shape[1]  # number of training examples
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) # Foreward propagation
    dw = (1/m) * np.dot(X, (A-Y).T)  # back propagation
    db = (1/m) * np.sum(A-Y)
    assert w.shape == dw.shape, print('mismatch between vecor and differenial. see tuple pairs: ', (w.shape, dw.shape))
    cost = np.squeeze(cost)
    grads = {'dw': dw,
             'db': db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """ This function optimizes w and b by running a gradient descent algorithm """
    costs = []    
    for i in range(num_iterations):
        # retrieve grads and cost
        grads, cost = propagate(w, b, X, Y)
        # retrieve parameters
        dw, db = grads['dw'], grads['db']
        # update parameters
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        # record cost for every 100th iteration
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    grads = {'dw':dw, 
            'db':db }
    params = {'w': w,
             'b': b}

    return grads, params, costs

def predict(w, b, X):
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)+ b)

    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    w, b = zero_param_init(X_train.shape[0])
    # Run Gradient descent 
    grads, params, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate)
    # Retrieve parameters w and b, after descent
    w, b = params['w'], params['b']
    # Predict test/train set examples
    Y_predict_test = predict(w, b, X_test)
    Y_predict_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_predict_test,
         "Y_prediction_train": Y_predict_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

# print_image()

#%%
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
standard_train_x = flatten_and_standardize(train_set_x_orig)
standard_test_x = flatten_and_standardize(test_set_x_orig)

d = model(standard_train_x, train_set_y, standard_test_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


#%%
