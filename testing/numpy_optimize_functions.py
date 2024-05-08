import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def preprocess_data(train_set_x_orig, test_set_x_orig):
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
        z = np.dot(w.T,X[:,i])+b  # shape(1,)
        z = z.squeeze()  # shape(1,) to shape()/scalar i.e. [0.0] to 0.0
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


def epoch_alt_for(X, Y, w, b, learning_rate=0.005):
    '''
    alternate way of writing epoch_for function. safer but slower.
    
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
    z = np.zeros((1,m))
    A = np.zeros_like(z)
    dz = np.zeros_like(z)
    dw = np.zeros((X.shape[0],1))
    db = 0
    cost=0
    for i in range(m):
        # FORWARD PROPAGATION
        z[0,i] = np.dot(w.T,X[:,i])+b
        A[0,i] = 1/(1+np.exp(-z[0,i]))

        # BACKWARD PROPAGATION (ADDING COST FUNCTION)
        cost = cost + (- (Y[0,i]*np.log(A[0,i])+(1-Y[0,i])*np.log(1-A[0,i])) )
        # BACKWARD PROPAGATION (ADDING GRADS)
        dz[0,i] = A[0,i]-Y[0,i]
        db = db + dz[0,i]
        # for n in range(X.shape[0]):    # double for loop, used in epoch_alt_2for
        #     dw[n,0] = dw[n,0] + X[n,i]*dz[0,i]
        dw[:,0] = dw[:,0] + X[:,i]*dz[0,i]

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
        z = np.dot(w.T,X[:,i])+b # shape(1,)
        z = z.squeeze()  # shape(1,) to shape()/scalar i.e. [0.0] to 0.0
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
    z = np.matmul(w.T,X)+b
    A = 1/(1+np.exp(-z))                                    # compute activation

    # BACKWARD PROPAGATION (FROM COST TO GRADs)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))        # compute cost
    cost = np.squeeze(np.array(cost))
    dz = A-Y
    dw = 1/m*np.matmul(X,dz.T)
    db = 1/m*np.sum(dz)

    # UPDATE PARAMETERS
    w = w - learning_rate*dw
    b = b- learning_rate*db
    
    return w, b, dw, db, cost


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using NEURON[learned logistic regression] parameters (w, b)
    
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
    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction


def logistic_regression(X, Y, num_epochs=2000, learning_rate=0.005, print_cost=False, epoch_fun=epoch):
    
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

