import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def hypothesis(self, X):
        #power of e
        e = np.exp(-np.dot(X, self.theta))

        #final formula
        hypothesis = 1 / (e + 1)

        return hypothesis

    def compute_cost(self, X, y, h):
        #inside the sigma
        formula = -(y * np.log(h) + ((1 - y) * np.log(1 - h)))

        #final formula
        J = np.sum(formula) / X.shape[0]
        
        return J

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        size_t = X.shape[1] + 1
        m = X.shape[0]

        # bias trick
        X = np.c_[np.ones((m, 1)), X]

        self.theta = np.random.rand(size_t)
        self.thetas.append(self.theta)
      
        
        for i in range(self.n_iter):
            h = self.hypothesis(X)
            J = self.compute_cost(X, y, h)
            
            # Stop the function when the difference between the previous cost and the current is less than eps
            if len(self.Js) > 0 and abs(self.Js[-1] - J) < self.eps:
                self.Js.append(J)
                break

            # Update the theta vector  
            self.theta -= self.eta * np.dot(X.T, h-y)
            self.thetas.append(self.theta)
            self.Js.append(J)

        
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        # bias trick
        m = X.shape[0]
        X = np.c_[np.ones((m, 1)), X]

        h = self.hypothesis(X)

        preds = (h >= 0.5).astype(int)

        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
  
    cv_accuracy = None
    cv_accuracies = []

    # set random seed
    np.random.seed(random_state)
    m = X.shape[0]

    ratio = np.random.permutation(m)
    fold = np.array_split(ratio, folds)

    for f in range(folds):
        
        # Split data into train and test sets.
        data_train = np.concatenate(fold[:f] + fold[f+1:])
        data_test = fold[f]

        X_train = X[data_train]
        y_train = y[data_train]

        X_test = X[data_test]
        y_test = y[data_test]
        
        # use lor model on the train and test sets
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)

        accuracy = np.sum(np.equal(y_pred, y_test)) / len(y_test)
        cv_accuracies.append(accuracy)

    cv_accuracy = np.mean(cv_accuracies)
    
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
   
    e = np.exp( -0.5 * ((data - mu) / sigma) ** 2)
    denominator = (sigma * np.sqrt(2 * np.pi))
    p = ((1 / denominator)  * e)
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.responsibilities = np.zeros((data.shape[0], self.k))
        self.weights = np.ones(self.k) / self.k
        self.mus = np.mean(np.array_split(data, self.k), axis=1)
        self.sigmas = np.std(np.array_split(data, self.k), axis=1)
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        weight = 0
        normal_response = [None] * self.k
        
        for d in range(self.k):
            delta_w = norm_pdf(data, self.mus[d], self.sigmas[d]) * self.weights[d]
            weight = weight + delta_w
            normal_response[d] = delta_w

        normal_response = normal_response / weight
        self.responsibilities = np.array(normal_response)

    def maximization(self, data):
        """
        M step - This function calculates and updates the model parameters
        """
        m = data.shape[0]

        # initialize fields
        update_weights = np.zeros(self.k)
        update_mus = np.zeros(self.k)
        update_sigmas = np.zeros(self.k)

        for d in range(self.k):
            update_weights[d] = np.sum(self.responsibilities[d]) / m
            w = update_weights[d] * m
            update_mus[d] = np.sum(self.responsibilities[d] * data) / w
            update_sigmas[d] = np.sqrt(np.sum(((data-update_mus[d])**2) * self.responsibilities[d] ) / w)
        
        # update fields
        self.weights = update_weights
        self.mus = update_mus
        self.sigmas = update_sigmas

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization functions to estimate the distribution parameters.
        Store the parameters in the attributes of the EM object.
        Stop the function when the difference between the previous cost and the current cost is less than the specified epsilon
        or when the maximum number of iterations is reached.

        Parameters:
        - data: The input data for training the model.
        """
        self.init_params(data)

        for i in range(self.n_iter):
          J = 0
          self.expectation(data)
          self.maximization(data)

          sum_pdf = np.sum([norm_pdf(data, self.mus[k], self.sigmas[k]) * self.weights[k] for k in range(self.k)], axis=0)
          J = np.sum([-np.log(np.sum(sum_pdf)) for k in range(self.k)], axis=0)
   
          self.costs.append(J)

          # Stop the function when the difference between the previous cost and the current is less than eps
          if i >= 1 and np.abs(self.costs[i-1] - J) < self.eps:
            break


    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.sum(norm_pdf(data, mus, sigmas) * weights)

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = None
        self.priors = {}
        self.gmm_params = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.classes = np.unique(y)
        
        for num_class in self.classes:
            data_class = X[np.where(y == num_class)]
            # Calculate the prior probability of the class
            self.priors[num_class] = len(data_class) / len(X)
            feature_m = []
            
            for feature in range(X.shape[1]):
                model = EM(k=self.k, random_state=self.random_state)
                model.fit(data_class[:, feature])
                feature_m.append(model)

            self.gmm_params[num_class] = feature_m     

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        for instance in X:
            prob_post = []

            for num_class in self.classes:
                likelihood = 1
                feature_models = self.gmm_params[num_class]
                
                for i in range(len(feature_models)):
                    likelihood = likelihood * gmm_pdf(instance[i], feature_models[i].weights, feature_models[i].mus, feature_models[i].sigmas)

                posterior = self.priors[num_class] * likelihood
                prob_post.append(posterior)
            max = np.argmax(prob_post)
            preds.append(max)

        return np.asarray(preds)

# Decision boundaries plot - taken from the notebook
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.array(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    from matplotlib import pyplot as plt
    
    # Logistic Regression
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)

    y_pred_train_lor = lor_model.predict(x_train)
    y_pred_test_lor = lor_model.predict(x_test)
    
    # calculate train and test accuracies
    lor_train_acc = np.mean(np.equal(y_train, y_pred_train_lor))
    lor_test_acc = np.mean(np.equal(y_test, y_pred_test_lor))

    # plot
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=lor_model, title="Logistic Regression Decision Boundaries")

    # Naive Bayes Gaussian
    naive_model = NaiveBayesGaussian(k=k)
    naive_model.fit(x_train, y_train)

    y_pred_train_naive = naive_model.predict(x_train)
    y_pred_test_naive = naive_model.predict(x_test)

    # calculate train and test accuracies
    bayes_train_acc = np.mean(np.equal(y_train, y_pred_train_naive))
    bayes_test_acc = np.mean(np.equal(y_test, y_pred_test_naive))

    # plot
    plt.figure()
    plot_decision_regions(x_train, y_train, classifier=naive_model, title="Naive Bayes Decision Boundaries")

    # Plot cost vs iteration number for Logistic Regression
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(lor_model.Js)), lor_model.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title("Cost VS Iteration Number for Logistic Regression Model")
    plt.grid(True)
    plt.show()
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    np.random.seed(1)

    a1 = np.concatenate((np.random.normal(7, 0.2, 200), np.random.normal(-7, 0.2, 200)))
    b1 = np.concatenate((np.random.normal(-7, 0.2, 200), np.random.normal(7, 0.2, 200)))

    a2 = np.concatenate((np.random.normal(-2 , 0.2 , 200), np.random.normal(-5 , 0.2 , 200)))
    b2 = np.concatenate((np.random.normal(-5 , 0.2 , 200), np.random.normal(-2 , 0.2 , 200)))

    a3 = np.concatenate((np.random.normal(-5 , 0.2 , 200) , np.random.normal(3 , 0.2 , 200)))
    b3 = np.concatenate((np.random.normal(-5 , 0.2 , 200) , np.random.normal(3 , 0.2 , 200)))
    
    dataset_a_features = (a1, a2, a3, b1, b2, b3)
    dataset_a_labels = (0, 0, 0, 1, 1, 1)
    
    c1 = np.concatenate((np.random.normal(0, 5, 200), np.random.normal(1, 5, 200)))
    d1 = np.concatenate((np.random.normal(-3, 5, 200), np.random.normal(-4, 5, 200)))

    c2 = c1 * 2
    d2 = d1 * 2

    c3 = np.concatenate((np.random.normal(0, 1, 200), np.random.normal(1, 1, 200)))
    d3 = np.concatenate((np.random.normal(-6, 1, 200), np.random.normal(-7, 1, 200)))
    
    dataset_b_features = (c1, c2, c3, d1, d2, d3)
    dataset_b_labels = (0, 0, 0, 1, 1, 1)
    
    
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }