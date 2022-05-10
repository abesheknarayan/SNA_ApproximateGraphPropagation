from data_processing import load_graph_dataset
import networkx as nx
import numpy as np

# constants for HKPR

e = 10**(-7) # margin of error
a = 0
b = 1
t = 5        # poisson distribution parameter
L = 5        # number of iterations
alpha = 0.85 # google's pagerank parameter


def page_rank_using_basic_propagation(graph,n):
    print('...starting PageRank using basic graph propagation...')
    m = graph.number_of_edges()
    print(n,m)

    X = np.array([1.0/n for i in range(n)])  # doubtful whether all are 1/n or only the seed node's value is 1/n
    
    W = np.array([alpha*(pow(1.0-alpha,i)) for i in range(n+1)]) # n+1 because we access Y[i+1] later edge condition

    Y = W.cumsum()

    r = np.array(X)   # initially r = x 
    r_ = np.zeros(r.shape) # extra temporary vector to store next level's r vector
    
    q = np.zeros(r.shape)

    pi = np.zeros(r.shape)

    for i in range(L):
        for u in graph.nodes:
            if r[u] != 0:
                for v in graph[u]:
                    r_[v] = r_[v] + ((Y[i+1]/Y[i])*(r[u]))/((pow(graph.degree[v],a))*(pow(graph.degree[u],b)))
                q[u] = (W[i]*r[u])/Y[i]
        pi = pi + q
        r = r_
        r_ = np.zeros(r.shape)
        q = np.zeros(r.shape)    

    q = np.array([(W[L]/Y[L])*r[i] for i in range(n)])
    pi = pi + q
    return pi



# def gnn_node_classification_using_random_propagation(graph):
#     print('...starting PageRank using randomized graph propagation...')

#     X = np.array([1.0/n for i in range(n)])  # doubtful whether all are 1/n or only the seed node's value is 1/n
    
#     W = np.array([alpha*(pow(1.0-alpha,i)) for i in range(n+1)]) # n+1 because we access Y[i+1] later edge condition

#     Y = W.cumsum()

#     r = np.array(X)   # initially r = x 
#     r_ = np.zeros(r.shape) # extra temporary vector to store next level's r vector
    
#     q = np.zeros(r.shape)

#     pi = np.zeros(r.shape)

#     for i in range(L):
#         for u in graph.nodes:
#             if r[u] != 0:
#                 for v in graph[u]:
#                     # extra condition
#                     if graph.degree[v] <= pow(((1/e)*(Y[i+1]/Y[i])*(r[u]/pow(graph.degree[u],b))),1/a):
#                         r_[v] = r_[v] + ((Y[i+1]/Y[i])*(r[u]))/((pow(graph.degree[v],a))*(pow(graph.degree[u],b)))
#                     else:
#                         # do subset sampling
                    

