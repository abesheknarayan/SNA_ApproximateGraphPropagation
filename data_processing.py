# file for loading graph from data

import networkx as nx


def load_graph_dataset(name="youtube"):
    print(f'...loading {name} graph... ')
    graph = nx.Graph()

    # hack to find the actual number of nodes with max node ID even though most of the nodes are singleton components
    max_node_id = 0

    with open(f'data/{name}_graph.txt') as f:
        for line in f:
          
          if line[0] == '#': 
              continue
          edge = tuple(map(int,line.split("\n")[0].split("\t")))

          # undirected graph
          graph.add_edge(edge[0]-1,edge[1]-1)
          max_node_id = max({max_node_id,edge[0],edge[1]})
    


    print(f'...completed loading {name} graph...')
    return graph,max_node_id

if __name__ == '__main__':
    graph = load_graph_dataset()