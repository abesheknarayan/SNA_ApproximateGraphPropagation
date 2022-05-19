from operator import attrgetter
from data_processing import load_graph_dataset
import networkx as nx
import numpy as np
from propagation_types import get_gnn_propagation, get_pagerank_propagation


def page_rank_using_basic_propagation(graph,n,L):
    print('...starting PageRank using basic graph propagation...')
    m = graph.number_of_edges()
    print(n,m)

    pagerank_parameters = get_pagerank_propagation(n,L)

    e,a,b,X,W,Y,alpha = attrgetter('e','a','b','X','W','Y','alpha')(pagerank_parameters)

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



def gnn_node_classification_using_random_propagation(graph,n,L):
    print('...starting GNN using randomized graph propagation...')

    gnn_parameters = get_gnn_propagation(n,L)
    e,a,b,X,W,Y,alpha = attrgetter('e','a','b','X','W','Y','alpha')(gnn_parameters)


    r = np.array(X)   # initially r = x 
    r_ = np.zeros(r.shape) # extra temporary vector to store next level's r vector
    
    q = np.zeros(r.shape)

    pi = np.zeros(r.shape)

    for i in range(L):
        for u in graph.nodes:
            if r[u] != 0:
                for v in graph[u]:
                    # extra condition
                    if graph.degree[v] <= pow(((1/e)*(Y[i+1]/Y[i])*(r[u]/pow(graph.degree[u],b))),1/a):
                        r_[v] = r_[v] + ((Y[i+1]/Y[i])*(r[u]))/((pow(graph.degree[v],a))*(pow(graph.degree[u],b)))
                    else:
                        # do subset sampling
                        ...
                q[u] = (W[i]*r[u])/Y[i]
        pi = pi + q
        r = r_
        r_ = np.zeros(r.shape)
        q = np.zeros(r.shape) 
        q = np.array([(W[L]/Y[L])*r[i] for i in range(n)])
        pi = pi + q
        return pi   


