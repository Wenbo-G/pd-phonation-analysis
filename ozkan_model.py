#The code here is based on the paper: A Comparison of Classification Methods for Telediagnosis of Parkinson’s Disease, Ozkan, 2016.
#DOI: 10.3390/e18040115
#https://www.mdpi.com/1099-4300/18/4/115
#It uses PCA to reduce the input space into principle components, and classifies this with the k-nearest-neighbour classifier


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import resultsHandler as rh

def ozkan(k,n_components,preprocessing_method,X_train,y_train,X_test,y_test):
    """The model: first applying PCA onto the feature space, then kNN on the components"""
    
    
    pca = PCA(n_components=n_components)
    
    if preprocessing_method == 'both': pca.fit(np.append(X_train,X_test,axis=0))
    else: pca.fit(X_train)
        
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    predicted_Y = clf.predict(X_test)
    prob_predicted_Y = clf.predict_proba(X_test)
    return rh.getPerformanceMetrics(y_test,predicted_Y,prob_predicted_Y[:, 1])
