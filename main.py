from pagerank import page_rank_using_basic_propagation
from data_processing import load_graph_dataset

import numpy as np

def main():
    graph,n = load_graph_dataset()
    pi = page_rank_using_basic_propagation(graph,n)
    print(len(np.unique(pi)))


if __name__ == '__main__':
    main()