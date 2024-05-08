import numpy as np
import h5py
from numba import njit


def load_dataset():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_X_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_Y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_X_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_Y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_Y_orig = train_Y_orig.reshape((1, train_Y_orig.shape[0]))
    test_Y_orig = test_Y_orig.reshape((1, test_Y_orig.shape[0]))
    
    return train_X_orig, train_Y_orig, test_X_orig, test_Y_orig, classes


def preprocess_data(train_set_x_orig, test_set_x_orig):
    """Reshape(flatten) and normalize the image data"""
    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape( train_set_x_orig.shape[0], -1 ).T
    test_set_x_flatten = test_set_x_orig.reshape( test_set_x_orig.shape[0], -1 ).T

    # Normalization
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    
    return train_set_x, test_set_x


def epoch_2for(X, Y, w, b, learning_rate=0.005):
    '''
    It uses 2 for loops; one to go through all samples and inner for loop to find gradient. HIGHLY UNOPTIMIZED! 
    Performs one epoch(cycle) of training for neuron[logistic regression]
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    learning_rate : float (default=0.05)   

    Returns
    -------
    w : updated array of learned weights by neuron
    b : updated learned bias by neuron
    dw : array containing increments that were added to weights
    db : increment that was added to bias
    cost: average loss of samples with input parameters
    '''    
    m = X.shape[1]
    dw = np.zeros((X.shape[0],1))
    db = 0
    cost=0
    for i in range(m):
        # FORWARD PROPAGATION
        z = np.dot(w.reshape(-1,),X[:,i])+b  # scalar/shape() --> (d,).(d,)
        # z = np.matmul(w.T,X[:,i])+b # shape(1,) --> (1,d)X(d,)
        # z = z.squeeze()  # shape(1,) to shape()/scalar i.e. [0.0] to 0.0
        A = 1/(1+np.exp(-z))

        # BACKWARD PROPAGATION (ADDING COST FUNCTION)
        cost = cost + (-( Y[0,i]*np.log(A)+(1-Y[0,i])*np.log(1-A)))
        # BACKWARD PROPAGATION (ADDING GRADS)
        dz = A-Y[0,i]
        for n in range(X.shape[0]):
            dw[n,0] = dw[n,0] + X[n,i]*dz
        db = db + dz

    # BACKWARD PROPAGATION (FINDING MEAN)
    cost = 1/m*cost
    dw = 1/m*dw
    db = 1/m*db

    # UPDATE PARAMETERS
    w = w - learning_rate*dw
    b = b- learning_rate*db
    
    return w, b, dw, db, cost


def epoch_for(X, Y, w, b, learning_rate=0.005):
    '''
    It uses for loop to go through all samples. NOT OPTIMIZED! 
    Performs one epoch(cycle) of training for neuron[logistic regression]
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    learning_rate : float (default=0.05)   

    Returns
    -------
    w : updated array of learned weights by neuron
    b : updated learned bias by neuron
    dw : array containing increments that were added to weights
    db : increment that was added to bias
    cost: average loss of samples with input parameters
    '''    
    m = X.shape[1]
    dw = np.zeros((X.shape[0],1))
    db = 0
    cost=0
    for i in range(m):
        # FORWARD PROPAGATION
        z = np.dot(w.reshape(-1,),X[:,i])+b  # scalar/shape() --> (d,).(d,)
        # z = np.matmul(w.T,X[:,i])+b # shape(1,) --> (1,d)X(d,)
        # z = z.squeeze()  # shape(1,) to shape()/scalar i.e. [0.0] to 0.0
        A = 1/(1+np.exp(-z))

        # BACKWARD PROPAGATION (ADDING COST FUNCTION)
        cost = cost + (-( Y[0,i]*np.log(A)+(1-Y[0,i])*np.log(1-A)))
        # BACKWARD PROPAGATION (ADDING GRADS)
        dz = A-Y[0,i]
        dw = dw + X[:,i].reshape(-1,1)*dz  # slicing make X[:,1] shape as (2,), need to convert as (2,1) so can add dw
        db = db + dz

    # BACKWARD PROPAGATION (FINDING MEAN)
    cost = 1/m*cost
    dw = 1/m*dw
    db = 1/m*db

    # UPDATE PARAMETERS
    w = w - learning_rate*dw
    b = b- learning_rate*db
    
    return w, b, dw, db, cost


def epoch(X, Y, w, b, learning_rate=0.005):
    '''
    It uses vectorization of numpy and no for loop to go through all samples. 
    OPTIMIZED! 
    Performs one epoch(cycle) of training for neuron[logistic regression]
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    learning_rate : float (default=0.05)   

    Returns
    -------
    w : updated array of learned weights by neuron
    b : updated learned bias by neuron
    dw : array containing increments that were added to weights
    db : increment that was added to bias
    cost: average loss of samples with input parameters
    '''    
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO A)
    z = np.matmul(w.T,X)+b    # shape(1,m) --> (1,d)X(d,m)
    A = 1/(1+np.exp(-z))      # compute activation; shape(1,m)

    # BACKWARD PROPAGATION (FROM COST TO GRADs)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))    # compute cost; shape()/scalar
    dz = A-Y    # shape(1,m)
    dw = 1/m*np.matmul(X,dz.T)    # shape(d,1) --> (d,m)X(m,1)
    db = 1/m*np.sum(dz)    # shape()/scalar

    # UPDATE PARAMETERS
    w = w - learning_rate*dw    # shape(d,1)
    b = b- learning_rate*db    # shape()/scalar
    
    return w, b, dw, db, cost


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using neuron[learned logistic regression] parameters (w, b)
    
    Parameters
    ----------
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data

    Returns
    -------
    Y_prediction: numpy.ndarray [shape: (1, #samples)]
        array containing predictions 0 or 1
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(-1, 1)
    
    # Compute forward propagation
    z = np.matmul(w.T,X)+b
    A = 1/(1+np.exp(-z))
    
    # Convert probabilities A[0,i] to actual predictions p[0,i]
    # for i in range(A.shape[1]):
    #     if A[0, i] > 0.5 :
    #         Y_prediction[0,i] = 1
    #     else:
    #         Y_prediction[0,i] = 0
    Y_prediction=(A >0.5).astype(int)
    
    return Y_prediction


def logistic_regression(X, Y, num_epochs=2000, learning_rate=0.005, print_cost=False, epoch_fun=epoch):
    """
    Single neuron flavoured logistic regression model that runs for num_epochs.
        Epoch_fun is passed as parameter to select which type of epoch function to use.
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    num_epochs : int (default=2000)
    learning_rate : float (default=0.005)
    print_cost : True/False (default=False)
        if True, it prints cost every 100 epochs and train accuracy
    epoch_fun : epoch function
        choices are [epoch, epoch_for, epoch_2for, epoch_numba]

    Returns
    -------
    d: dictionary {Y_prediction_train" : Y_prediction, costs, w, b,
         dw, db, learning_rate, num_epochs}
    """
    ## initialize parameters with zeros
    w = np.zeros((X.shape[0],1))
    b = 0.0

    ## epochs loop
    costs = []
    for i in range(num_epochs):
        # Cost and gradient calculation 
        w, b, dw, db, cost = epoch_fun(X, Y, w, b, learning_rate)
           
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training epochs
            if print_cost:
                print(f"Cost after epochs {i}, {cost}")
    
    ## Predict test/train set examples
    Y_prediction = predict(w,b,X)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))

    
    d = { 
         "Y_prediction_train" : Y_prediction, 
         "costs": costs, "w" : w, "b" : b,
         "dw" : dw, "db" : db,
         "learning_rate" : learning_rate, "num_epochs": num_epochs}
    
    return d


@njit
def epoch_numba(X, Y, w, b, learning_rate=0.005):
    '''
    It uses numba and no for loop to go through all samples. 
    OPTIMIZED! 
    Performs one epoch(cycle) of training for neuron[logistic regression]
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    learning_rate : float (default=0.05)   

    Returns
    -------
    w : updated array of learned weights by neuron
    b : updated learned bias by neuron
    dw : array containing increments that were added to weights
    db : increment that was added to bias
    cost: average loss of samples with input parameters
    '''    
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO A)
    z = np.dot(w.T,X)+b
    A = 1/(1+np.exp(-z))                                    # compute activation

    # BACKWARD PROPAGATION (FROM COST TO GRADs)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))        # compute cost
    dz = A-Y
    dw = 1/m*np.dot(X,dz.T)
    db = 1/m*np.sum(dz)

    # UPDATE PARAMETERS
    w = w - learning_rate*dw
    b = b- learning_rate*db
    
    return w, b, dw, db, cost


@njit
def predict_numba(w, b, X):
    '''
    numba optimized.
    Predict whether the label is 0 or 1 using neuron[learned logistic regression] parameters (w, b)
    
    Parameters
    ----------
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data

    Returns
    -------
    Y_prediction: numpy.ndarray [shape: (1, #samples)]
        array containing predictions 0 or 1
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(-1, 1)
    
    # Compute forward propagation
    z = np.dot(w.T,X)+b
    A = 1/(1+np.exp(-z))
    
    # Convert probabilities A[0,i] to actual predictions p[0,i]
    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction


@njit
def logistic_regression_numba(X, Y, num_epochs=2000, learning_rate=0.005, print_cost=False, epoch_fun=epoch_numba):
    """
    Single neuron flavoured logistic regression model that runs for num_epochs.
        Epoch_fun is passed as parameter to select which type of epoch function to use.
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    num_epochs : int (default=2000)
    learning_rate : float (default=0.005)
    print_cost : True/False (default=False)
        if True, it prints cost every 100 epochs and train accuracy
    epoch_fun : epoch_numba function
    """
    ## initialize parameters with zeros
    w = np.zeros((X.shape[0],1))
    b = 0.0

    ## epochs loop
    costs = []
    for i in range(num_epochs):
        # Cost and gradient calculation 
        w, b, dw, db, cost = epoch_fun(X, Y, w, b, learning_rate)
           
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training epochs
            if print_cost:
                # print(f"Cost after epochs {i}, {cost}")
                print("Cost after epochs ", i," : ", cost)
    
    ## Predict test/train set examples
    Y_prediction = predict_numba(w,b,X)

    # Print train/test Errors
    if print_cost:
        # print(f"train accuracy: {100 - np.mean(np.abs(Y_prediction - Y)) * 100}")
        print("train accuracy: ",100 - np.mean(np.abs(Y_prediction - Y)) * 100)

    # TODO: returning dictionary in numba workaround
    # d = { 
    #      "Y_prediction_train" : Y_prediction, 
    #      "costs": costs, "w" : w, "b" : b,
    #      "dw" : dw, "db" : db,
    #      "learning_rate" : learning_rate, "num_epochs": num_epochs}
    
    # return d



