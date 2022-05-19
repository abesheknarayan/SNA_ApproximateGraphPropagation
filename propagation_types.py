import numpy as np
from propagation_parameters import PropagationParameters

def get_pagerank_propagation(n,L):
    e = 10**(-7)
    a = 0
    b = 1
    X = np.array([1.0/n for i in range(n)])
    alpha = 0.85 # Google Search Parameter
    W = np.array([alpha*(pow(1.0-alpha,i)) for i in range(n+1)]) 
    Y = W.cumsum()
    

    return PropagationParameters(n,e,a,b,X,W,Y,L,None,alpha)

def get_gnn_propagation(n,L):
    e = 10**(-7)
    a = 0.5
    b = 0.5
    alpha = 0.85
    W = np.array([alpha*(pow(1.0-alpha,i)) for i in range(n+1)])
    X = np.array([1 for i in n])
    Y = W.cumsum()

    return PropagationParameters(n,e,a,b,X,W,Y,L,None,alpha)