###### Your ID ######
# ID1: 318247061
# ID2: 211357751
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X_norm = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0)) #mean = avg of the array elements.
    y_norm = (y - np.mean(y)) / (np.max(y) - np.min(y))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X_norm, y_norm

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    
    X = np.c_[np.full(X.shape[0],1), X]
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    #h = np.dot(X, theta)  
    #error = h - y
    #J = (1 / (2 * m)) * np.sum(np.square(error))  # mean squared error cost

    
    J = 0  # We use J for the cost.
    h = (np.dot(X, theta)) # hypothesis
    err = h - y
    m = X.shape[0]  # number of training examples
    J = (1 / (2 * m)) * np.sum(np.dot(err, err))
    return J
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = X.shape[0]

    for i in range(num_iters):
        h = (np.dot(X, theta)) # hypothesis
        err = h - y
        gradient = np.dot(X.T, err) / m ##derivative
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return pinv_theta 


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    m = X.shape[0]
    for i in range(num_iters):
        h = (np.dot(X, theta)) # hypothesis
        err = h - y
        gradient = np.dot(X.T, err) / m ##derivative
        theta = theta - alpha * gradient
        J = compute_cost(X, y, theta)
        
        # Stop gradient descent when cost is too high
        if J > 1e10:
            break

        J_history.append(J)

        # Stop when improvement is smaller than 1e-8
        if i > 0 and abs(J_history[-1] - J_history[-2]) < 1e-8:
            break
    
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    for i in alphas:
        theta = np.ones(X_train.shape[1])
        theta, _ = efficient_gradient_descent(X_train, y_train, theta, i, iterations) 
        validation_loss = compute_cost(X_val, y_val, theta)
        alpha_dict[i] = validation_loss

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    np.random.seed(42)

    for i in range(5):
        best_feature = None
        best_validation_loss = float("inf")
        theta = np.random.random(size = i + 2)

        for feature in remaining_features:
            current_features = selected_features + [feature]

            X_train_selected = X_train[:, current_features]
            X_val_selected = X_val[:, current_features]

            # Add bias to the selected feature set
            X_train_selected_bias = np.hstack([np.ones((X_train_selected.shape[0], 1)), X_train_selected])
            X_val_selected_bias = np.hstack([np.ones((X_val_selected.shape[0], 1)), X_val_selected])

            #theta = np.zeros(X_train_selected_bias.shape[1])
            theta, _ = efficient_gradient_descent(X_train_selected_bias, y_train, theta, best_alpha, iterations)
            validation_loss = compute_cost(X_val_selected_bias, y_val, theta)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_feature = feature

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features    # return the list of selected feature indices
    
   

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
   # Compute the square of each feature and interaction terms
    n = len(df.columns)
    for i in range(n):
        for j in range(i, n):
            # Compute the product of the feature pair
            poly_feature = df.iloc[:, i] * df.iloc[:, j]

            # Create a new column name for the polynomial feature
            if i == j:
                poly_feature_name = f"{df.columns[i]}^2"
            else:
                poly_feature_name = f"{df.columns[i]} * {df.columns[j]}"

            # Add the polynomial feature to the dataframe
            df_poly[poly_feature_name] = poly_feature

    return df_poly
