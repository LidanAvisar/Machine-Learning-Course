import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]
    while np.unique(centroids).shape[0] < k:
        random_indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[random_indices]
    
    return np.asarray(centroids).astype(np.float64)

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    rowX = X.reshape((1, X.shape[0], 3))
    colCentroids = centroids.reshape((centroids.shape[0], 1, 3))
    d = rowX - colCentroids
    distances = np.sum(np.abs(d) ** p, axis=2)
    prob = 1 / p
    distances = np.power(distances, prob)

    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)

    for i in range(max_iter):
        minimumCentroids = np.argmin(lp_distance(X, centroids, p), axis=0)
        newCentroids = np.empty((k, 3))

        for j in range(k):
            mean = np.mean(X[minimumCentroids == j], axis=0)
            newCentroids[j] = mean

        if np.array_equal(centroids, newCentroids):
            break

        centroids = newCentroids
        
    classes = minimumCentroids

    return centroids, classes


def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None

    centroids = get_random_centroids(X, 1)
    X_copy = np.copy(X)
    deleteRow = np.any(X_copy == np.array(centroids[0]).reshape(1, -1), axis=1)
    X_copy = X_copy[deleteRow == False]
    
    while len(centroids) < k:
        distance = np.min(lp_distance(X_copy, centroids, p), axis=0)
        distance = distance ** 2
        distance = distance / np.sum(distance)
        idx = np.random.choice(X_copy.shape[0], 1, p=distance)
        newCent = X_copy[idx, :]
        deleteRow = np.any(X_copy == np.array(newCent).reshape(1, -1), axis=1)
        X_copy = X_copy[deleteRow == False]
        centroids = np.vstack((centroids, newCent))
        
    for i in range(max_iter):
        minimumCentroids = np.argmin(lp_distance(X, centroids, p), axis=0)
        new_centroids = np.empty((k, 3))
        
        for j in range(k):
            new_centroids[j] = np.mean(X[minimumCentroids == j], axis=0)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids
        
    classes = minimumCentroids

    return centroids, classes
