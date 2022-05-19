import time
from pagerank import page_rank_using_basic_propagation
from data_processing import load_graph_dataset
from secure.test_pagerank import testScatterAndGatherContactTracing
import matplotlib.pyplot as plt

import numpy as np

def main():
    graph,n = load_graph_dataset()
    pagerank_metric_x = []
    pagerank_metrix_y = []
    for i in [1,2,3,4]:
        st_time = time.time()
        pi = page_rank_using_basic_propagation(graph,n,i)
        en_time = time.time()
        print(i,en_time-st_time)
        pagerank_metric_x.append(10*i)
        pagerank_metrix_y.append(en_time-st_time)
    plt.plot(pagerank_metric_x,pagerank_metrix_y)
    plt.xlabel("No of iterations")
    plt.ylabel("Time (in seconds)")
    plt.savefig('output/pagerank_metric.png')

    # print(len(np.unique(pi)))

    print("...doing secure pagerank...")

    testScatterAndGatherContactTracing()



if __name__ == '__main__':
    main()