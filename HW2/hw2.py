from itertools import count
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 1
    
    s = np.count_nonzero(data[:,-1])

    for d in np.unique(data[:,-1], return_counts=True)[1]:
        gini -= (d / s) ** 2

    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    s = np.count_nonzero(data[:,-1])

    for d in np.unique(data[:,-1], return_counts=True)[1]:
        si = d / s
        entropy -= si * math.log(si, 2)
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = impurity_func(data)
    groups = {} # groups[feature_value] = data_subset
    s = np.count_nonzero(data[:, feature])
    instances = data.shape[0]
    sum = 0

    for value in np.unique(data[:, feature]):
        v = np.count_nonzero(data[:, feature] == value) 
        goodness -=  (v / s) * impurity_func(data[data[:, feature] == value])
        groups[value] =  data[data[:, feature] == value]

    if gain_ratio:
        featureCol = data[:,feature]
        _, counts = np.unique(featureCol, return_counts=True)
        instances = data.shape[0]
        countsLog = np.log2(counts / instances)
        sum = np.dot(counts, countsLog) / instances
        
        splitInformation = 0 - sum
        
        if splitInformation != 0:
            goodness /= splitInformation
        else:
            goodness = 0
    
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        
        label, count = np.unique(self.data[:, -1], return_counts=True)
        return label[np.argmax(count)]
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        max = float('-inf')
        best_f = -1

        for f in range(self.data.shape[1] - 1):
            imp_f, temp_groups = goodness_of_split(self.data, f, impurity_func, self.gain_ratio) #self.data[:, f-1:f]
            if imp_f > max:
                max = imp_f
                groups = temp_groups
                best_f = f
        
        self.feature = best_f
        labels, counts = np.unique(self.data[:, -1], return_counts=True)
        f_vals = np.unique(self.data[:, self.feature])
    
        if self.terminal or len(labels) == 1 or len(f_vals) == 1 or self.depth >= self.max_depth:
            self.terminal = True
            return

        py0 = counts[0] / self.data.shape[0] #edible / rows 
        py1 = counts[1] / self.data.shape[0] #poisionable / rows        
        chiF = 0
            
        for v in f_vals: # run on vals array
            dataChild = self.data[self.data[:, self.feature] == v, :] 
            
            df = dataChild.shape[0]
            labels_v, pf = (np.unique(dataChild[:,-1], return_counts=True))
            pf0 = pf[0]

            if pf.shape[0] > 1:
                pf1 = pf[1]
            
            if len(labels_v) > 1:
                nf = pf1
            elif labels_v[0] == labels[0]:
                nf = 0
            else:
                nf = pf0
                pf0 = 0
                    
            e0 = df * py0
            e1 = df * py1
            chiF += ((((pf0-e0)**2)/e0) + (((nf-e1)**2)/e1))

        if self.chi < 1 and len(f_vals) > 1:
            if chiF <= chi_table[len(f_vals) - 1][self.chi]:
                self.terminal = True
                return
      
        for value in groups:
            #v_data = self.data[self.data[:, best_f] == value]
            self.add_child(DecisionNode(groups[value], -1, self.depth + 1, self.chi, self.max_depth, self.gain_ratio), value)



def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)

    q = [root]
    while q:
        current = q.pop()

        if not current.terminal:
            current.split(impurity)
            for child in current.children:
                q.append(child)
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    while not root.terminal:
        feature_value = instance[root.feature]
        found_child = False
        for i, child in enumerate(root.children):
            if root.children_values[i] == feature_value:
                root = child
                found_child = True
                break
        if not found_child:
            break
    return root.pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    counter = 0

    for instance in dataset:
        if predict(node, instance) == instance[-1]:
            counter += 1

    accuracy = (counter / dataset.shape[0])

    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, calc_entropy, gain_ratio=True, chi=1, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
    return training, testing

def calc_chi(node):
    chi_squared = 0.0
    P_Y = np.unique(node.data[:, -1], return_counts=True)[1]
    P_Y = P_Y / len(node.data)
    for val in node.children_values:
        f_data = node.data[node.data[:, node.feature] == val]
        D_f = len(f_data)
        p_f = np.unique(f_data[:, -1], return_counts=True)[1]
        E = D_f*P_Y
        chi_squared += np.sum(np.divide(np.square(p_f-E), E))
    return chi_squared

def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []  

    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train, calc_entropy, gain_ratio=True, chi=chi)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))

        max = 0
        q = [tree]
        while q:
            current = q.pop()

            if current.depth > max:
                max = current.depth
                
            for child in current.children:
                q.append(child)
                
        depth.append(max)

    return chi_training_acc, chi_testing_acc, depth



def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None

    if node is None:
        n_nodes = 0
    else:
        n_nodes = 1
        for child in node.children:
            n_nodes += count_nodes(child)

    return n_nodes

