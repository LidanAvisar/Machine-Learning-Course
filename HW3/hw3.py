from pydoc import classname
from statistics import mean
import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        
        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.4,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.4
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.4,
            (0, 1): 0.1,
            (1, 0): 0.1 ,
            (1, 1): 0.4
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.3,
            (0, 0, 1): 0.1,
            (0, 1, 0): 0.2,
            (0, 1, 1): 0.1,
            (1, 0, 0): 0.1,
            (1, 0, 1): 0.4,
            (1, 1, 0): 0.1,
            (1, 1, 1): 0.1
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        pX = list(X.values())        
        pY = list(Y.values())
        pX_Y = list(X_Y.values())
        pXmultY = []

        for x in pX:
            for y in pY:
                pXmultY.append(x * y)
        
        return not pX_Y == pXmultY


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        pX_Y_C = list(X_Y_C.values())        
        pX_C = list(X_C.values())
        pY_C = list(Y_C.values())
        pC = list(C.values())
        
        pX_YgivenC = []
        pXgivenC = []
        pYgivenC = []
        mult = []

        for xyc in pX_Y_C:
            for c in pC:
                pX_YgivenC.append(xyc / c)
        
        #print(pX_YgivenC)
                
        for xc in pX_C:
            for c in pC:
                pXgivenC.append(xc / c)
        
        for yc in pY_C:
            for c in pC:
                pYgivenC.append(yc / c)
        
        for xc in pXgivenC:
            for yc in pYgivenC:
               mult.append(xc * yc)

        return not pX_YgivenC == mult      

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    return np.log2(((rate ** k) * np.exp(-rate)) / np.math.factorial(k)) #CHECKKKKKKKKKKKKKK

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []

    #for i, j in zip(rates, samples):
    #   likelihoods.append(np.sum(poisson_log_pmf(j, i)))

    for rate in rates:
        for i in samples:
            mult = np.sum(poisson_log_pmf(i, rate))
        likelihoods.append(mult)

    return likelihoods

    #for rate in rates:
    #    likelihoods.append(np.sum(poisson_log_pmf(samples, rate)))
    

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """

    max_likelihood = float('-inf')
    best_rate = None

    for rate in rates:
        log_likelihood = 0.0
        for sample in samples:
            log_likelihood += poisson_log_pmf(sample, rate)
        if log_likelihood > max_likelihood:
            max_likelihood = log_likelihood
            best_rate = rate

    return best_rate

  
def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    sum = 0
    n = len(samples)

    for sample in samples:
        sum += sample
    
    mean = sum / n
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """

    frac = 1 / (np.sqrt(2 * np.pi * std * std))
    pow = ((x - mean) / std) ** 2
    e = np.exp(-0.5 * pow)
    p = frac * e

    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
       
        #self.mean = np.mean(dataset[dataset[:, -1] == class_value, :-1], axis=0)
        #self.std = np.std(dataset[dataset[:, -1] == class_value, :-1], axis=0)
        self.class_value = class_value
        self.dataset = dataset
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        n = self.dataset.shape[0]
        count = np.sum(self.dataset[:, -1] == self.class_value)
        prior = count / n

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = x / self.get_prior()
      
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        
        post0 = self.ccd0.get_instance_posterior(x)
        post1 = self.ccd1.get_instance_posterior(x)
        if np.max(post1) > np.max(post0):
            return 1
       
        return 0
 

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    
    num_correct = 0
    for instance in test_set:
        x = instance[:-1]
        y = instance[-1]
        y_pred = map_classifier.predict(x)
        if y_pred == y:
            num_correct += 1
    acc = num_correct / len(test_set)

    return acc 
    
    

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    
    d = len(x)
    det = np.linalg.det(cov)
    invers = np.linalg.inv(cov)
    pdf = np.exp(-0.5 * (x - mean).transpose() @ invers @ (x - mean)) / np.sqrt(((2 * np.pi) ** d) * det)

    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.dataset_by_class = dataset[dataset[:,-1] == class_value] 
        self.features_by_class = self.dataset_by_class[:, :-1]
        self.mean = np.mean(self.features_by_class, axis = 0) 
        self.cov_matrix = np.cov(self.features_by_class, rowvar = False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        n = self.dataset.shape[0]
        count = np.sum(self.dataset[:, -1] == self.class_value)
        prior = count / n

        return prior
     
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
   
        return multi_normal_pdf(x, self.mean, self.cov_matrix)
        
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior
      

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        post0 = self.ccd0.get_prior()
        post1 = self.ccd1.get_prior()
        if np.max(post1) > np.max(post0):
            return 1
        
        return 0

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        post0 = self.ccd0.get_instance_likelihood(x)
        post1 = self.ccd1.get_instance_likelihood(x)
        if np.max(post1) > np.max(post0):
            return 1
        
        return 0

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.dataset_by_class = dataset[dataset[:,-1] == class_value] 
        self.features_by_class = self.dataset_by_class[:, :-1]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        n = self.dataset.shape[0]
        count = np.sum(self.dataset[:, -1] == self.class_value)
        prior = count / n

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        n_i = len(self.dataset_by_class)
        likelihood = 1
        features = len(x) - 1
        
        for feature in range(features):
            n_ij_values, n_ij_counts = np.unique(self.dataset_by_class[:,feature], return_counts = True)
            v_j = len(np.unique(self.dataset[:,feature]))
            
            index = np.where(n_ij_values == x[feature])
            if(len(index[0]) == 0):
                n_ij = EPSILLON
            else:
                n_ij = n_ij_counts[index]
            
            likelihood *= (n_ij + 1) / (n_i + v_j)
        
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        post0 = self.ccd0.get_instance_posterior(x)
        post1 = self.ccd1.get_instance_posterior(x)
        if np.max(post1) > np.max(post0):
            return 1
       
        return 0

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        correctly_classified = 0
        
        for instance in test_set:
            if self.predict(instance) == instance[-1]:
                correctly_classified+=1

        return (correctly_classified / len(test_set))

