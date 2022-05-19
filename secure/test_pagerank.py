from ctypes import addressof
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import torch
import sys
import networkx as nx
import matplotlib.pyplot as plt

import math
  
int_infinity = 100000000  
# adding components to the system path

import secure.bitonic_sort as bitonic_sort
import secure.secure_pagerank as secure_pagerank
import secure.graph_structure as graph_structure


#Initialise Crypten
crypten.init()

torch.set_num_threads(1)


def nextPowerOf2(n):
    count = 0
 
    # First n in the below
    # condition is for the
    # case where n is 0
    if (n and not(n & (n - 1))):
        return n
     
    while( n != 0):
        n >>= 1
        count += 1
     
    return 1 << count

#Graph List structure (u, v, isVertex, data, covid risk, rho, residue, degree)
@mpc.run_multiprocess(world_size=2)
def testScatterAndGatherContactTracing():

    arr_used = [[1, 1, 1, 0, 0, 0, 1/4, 1, 0, 0], [2, 2, 1, 0, 3, 0, 1/4, 2, 0, 0], [3, 3, 1, 0, 0, 0, 1/4, 3, 0, 0], [4, 4, 1, 0, 0, 0, 1/4, 2, 0, 0], [1, 3, 0, 0, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, 0, 0, 0], [3, 4, 0, 0, 1, 0, 0, 0, 0, 0], [4, 3, 0, 0, 1, 0, 0, 0, 0, 0], [4, 2, 0, 0, 0, 0, 0, 0, 0, 0], [2, 4, 0, 0, 0, 0, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0, 0, 0, 0], [3, 2, 0, 0, 0, 0, 0, 0, 0, 0]]
    G = nx.Graph()
    for edge in arr_used:
        if edge[0] != edge[1]:
            G.add_edge(str(edge[0]),str(edge[1]))

    nx.draw(G)
    plt.savefig('output/secure_graph.png',with_labels=True)
        
    nearestPowerOf2 = nextPowerOf2(len(arr_used))

    
    for i in range(0, nearestPowerOf2 - len(arr_used)):
        arr_used.append([int_infinity, int_infinity,int_infinity, int_infinity, int_infinity, int_infinity, int_infinity, int_infinity, int_infinity, int_infinity])
    for i in range(0, len(arr_used)): 
        arr_used[i][8] = int_infinity
    
    arr_used[0][8] = 0

    a_enc = crypten.cryptensor(arr_used, ptype=crypten.mpc.arithmetic)

    scatterAndGatherContactPageRankObject = secure_pagerank.ScatterAndGatherPageRank(len(arr_used), 0.15, 0)
    graphStructureObject = graph_structure.EdgeListEncodedGraph(len(arr_used), a_enc, scatterAndGatherContactPageRankObject)

    L = 1

    for i in range(0, L):
        graphStructureObject.performScatter()
        graphStructureObject.performGather()

    rank = comm.get().get_rank()
    afterGraph = graphStructureObject.graphList.get_plain_text()
    if rank == 0:
        for i in range(0, len(afterGraph)):
            if afterGraph[i][0] == afterGraph[i][1]:
                print(afterGraph[i][0], afterGraph[i][1], afterGraph[i][5])

