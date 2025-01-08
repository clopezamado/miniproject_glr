import networkx as nx
import random
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric
import matplotlib.pyplot as plt
import torch_geometric.datasets as datasets
from torch_geometric.utils import to_networkx, from_networkx
import math


def obtainSparseGraph(numNodes, levelSparsity, numlabels):
    """
    Generate a connected graph with a specified sparsity level.
    
    Parameters:
    - numNodes: Number of nodes in the graph.
    - levelSparsity: A float between 0 and 1.
      * 1: Tree -> minimum edges to remain connected (numNodes - 1).
      * 0: Fully connected graph.
      * Intermediate values generate a graph with a proportional number of edges.
    - numLabels: Number of distinct labels
    - train_per_class: Number of training nodes per class (default: 20)
    - val_ratio: Ratio of nodes used for validation (default: 15%).
    - test_ratio: Ratio of nodes used for testing (default: 40%)
    
    Returns:
    - G: PyTorch Geometric data object -> connected graph with the specified sparsity level.
    """

    # Start with a tree to ensure connectivity
    G = nx.random_tree(numNodes)    
    # Calculate the number of edges based on sparsity level
    maxEdges = numNodes * (numNodes - 1) // 2  # Maximum edges for a complete graph
    minEdges = numNodes - 1  # Minimum edges for a tree
    numEdges = int(minEdges + (1 - levelSparsity) * (maxEdges - minEdges))
    
    # Add additional edges to meet the desired sparsity level
    edges_to_add = numEdges - minEdges
    all_possible_edges = list(nx.non_edges(G))
    random.shuffle(all_possible_edges)
    
    for u, v in all_possible_edges[:edges_to_add]:
        G.add_edge(u, v)   
    data = from_networkx(G)
    data.x = torch.rand((numNodes,128))
    data.y = torch.randint(0, numlabels, (numNodes,))
    
    return data
def obtainUniformGraphWithHubs(num_nodes, num_edges,num_hubs,num_labels):
    if num_hubs!=0:
        if num_hubs==1:
            G = nx.Graph()
            hub = random.choice(range(num_nodes))
            for node in list(set(range(num_nodes)) - {hub}):
                G.add_edge(hub,node)
        else:
            #Create cyclic graph between hubs
            G = nx.cycle_graph(num_hubs)
            num_nodes_per_hub=int(num_nodes/num_hubs)-1
            extra_nodes= num_nodes%num_hubs   
            new_node_counter = max(G.nodes) + 1
            #Create hubs
            for node in list(G.nodes):
                for _ in range(num_nodes_per_hub):
                    G.add_edge(node, new_node_counter)
                    new_node_counter += 1
                if node<extra_nodes:
                    G.add_edge(node, new_node_counter)
                    new_node_counter += 1    
            
    else:
        G = nx.cycle_graph(num_nodes)
        #G = nx.path_graph(num_nodes)
        
    rest_edges=num_edges-len(G.edges)
    all_possible_edges = list(nx.non_edges(G))
    random.shuffle(all_possible_edges)    
    for u, v in all_possible_edges[:rest_edges]:
        G.add_edge(u, v)
    #nx.draw(G, with_labels=True, font_weight='bold', node_color='lightblue', edge_color='gray')
    plt.show()     
    data = from_networkx(G)
    data.x = torch.rand((num_nodes,128))
    data.y = torch.randint(0, num_labels, (num_nodes,))
    #draw_graph(data)
        
    return data 
def obtainUniformGraphWithHubs_regular(num_nodes, num_edges,num_hubs,num_labels):
    if num_hubs!=0:
        #Start by creating a spanning tree (connected graph with n-1 edges)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = list(nx.generators.trees.random_tree(num_nodes).edges())
        G.add_edges_from(edges)
        
        # Remaining edges to distribute
        remaining_edges = num_edges - (num_nodes - 1)
        #Select the hubs and distribute the remaining edges
        hubs = random.sample(range(num_nodes), num_hubs)
        #1/(num_hubs) edges for each hub
        edges_per_hub = remaining_edges // (num_hubs)
        rest_edges = remaining_edges - edges_per_hub*num_hubs  # Any leftover edges will be distributed
        # Add edges to hubs first
        for hub in hubs:
            # Create a set of neighbors to connect to the hub
            possible_neighbors = set(range(num_nodes)) - {hub}
            added_edges = 0
            while added_edges < edges_per_hub and possible_neighbors:
                neighbor = random.choice(list(possible_neighbors))
                if not G.has_edge(hub, neighbor):  # Check if the edge already exists
                    G.add_edge(hub, neighbor)
                    added_edges += 1
                possible_neighbors.remove(neighbor)  # Remove the neighbor from the set
                
            rest_edges+=(edges_per_hub-added_edges)#If there are no more possible edges to add to hub, we will distribute them uniformly in the graph    
        #distribute the rest of the edges randomly
        all_possible_edges = list(nx.non_edges(G))
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges[:rest_edges]:
            G.add_edge(u, v)
    else:
        d = (2 * num_edges) // num_nodes
        G=nx.random_regular_graph(d, num_nodes)
        current_edges = G.number_of_edges()
        if num_edges>current_edges:
            add_edges=num_edges-current_edges
            all_possible_edges = list(nx.non_edges(G))
            random.shuffle(all_possible_edges)
            for u, v in all_possible_edges[:add_edges]:
                G.add_edge(u, v)
        elif num_edges<current_edges:
            connected=False
            while not connected:
                remove_edges=current_edges-num_edges
                all_possible_edges=list(G.edges())
                random.shuffle(all_possible_edges)
                for u, v in all_possible_edges[:add_edges]:
                    G.remove_edge(u, v)
                connected=G.is_connected()
                        
    data = from_networkx(G)
    data.x = torch.rand((num_nodes,128))
    data.y = torch.randint(0, num_labels, (num_nodes,))

    #draw_graph(data)
    return data  
    
# Function to create a spanning tree from the graph
def get_spanning_tree(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.numpy().T)
    tree = list(nx.minimum_spanning_edges(G, algorithm='kruskal', data=False))
    tree_edges = torch.tensor(tree).t().contiguous()
    return tree_edges

def apply_sparsity(data,sparsity):
    
    edge_index=data.edge_index
    num_nodes=data.num_nodes
    
    # Create a graph from the original edge_index
    G = nx.Graph()
    G.add_edges_from(edge_index.numpy().T)
    
    # Generate the spanning tree (s = 1 case)
    spanning_tree_edges = get_spanning_tree(edge_index)
    
    max_edges = num_nodes * (num_nodes - 1) // 2  # Total possible edges
    add_edges = int((1 - sparsity) * (max_edges-(num_nodes-1))) #number of edges to add to spanning tree
    total_edges=num_nodes-1+add_edges
    
    current_edges=list(G.edges())
    edges_to_keep = set(tuple(edge) for edge in spanning_tree_edges.t().tolist())
    
    # Remove random edges if too many
    if len(current_edges)>total_edges:
        num_edges_to_remove=len(current_edges)-total_edges
        removable_edges = list(set(current_edges) - edges_to_keep)
        random.shuffle(removable_edges)
        edges_to_keep.update(removable_edges[:len(removable_edges)-num_edges_to_remove])
    elif len(current_edges) < total_edges:
        potential_edges = set(nx.non_edges(G))  # All possible edges not currently in the graph
        new_edges = random.sample(potential_edges, total_edges - len(current_edges))
        edges_to_keep.update(new_edges)
    final_edges = torch.tensor(list(edges_to_keep)).t().contiguous()
    print("sparse: "+str(final_edges.size()))
    return final_edges

def draw_graph(data, node_size=300, font_size=10, with_labels=True):
    """
    Draws a graph from a PyTorch Geometric Data object.
    
    Parameters:
    - data (torch_geometric.data.Data): The graph data to visualize.
    - node_labels (list or None): Optional node labels to display.
    - node_size (int): Size of the nodes in the plot.
    - font_size (int): Font size for the labels.
    - with_labels (bool): Whether to show node labels.
    """
    # Convert PyTorch Geometric data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Plot the graph
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_size=node_size,
        font_size=font_size,
        node_color="skyblue",
        edge_color="gray",
    )
    plt.title("Graph Visualization")
    plt.show()

def obtainDiameterNodes(G):
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    # Find the pair of nodes with the maximum shortest path
    max_distance = 0
    node_pair = None
    for u, lengths in shortest_paths.items():
        for v, distance in lengths.items():
            if distance > max_distance:
                max_distance = distance
                node_pair = (u, v)
    return max_distance,node_pair    

def changeDiameter(data,steps):
    #G1 changes diameter in each step, G2 adds edge but doesn't change diameter, G3 adds edge to change slightly ASPL
    #draw_graph(data)
    G1=to_networkx(data, to_undirected=True)
    G2=G1.copy()
    diameters=[]
    graphs_changedDiameters=[]
    graphs_noChangedDiameters=[]
    data_aux=from_networkx(G1)
    data_aux.x=data.x
    data_aux.y=data.y
    graphs_changedDiameters.append(data_aux)
    data_aux=from_networkx(G2)
    data_aux.x=data.x
    data_aux.y=data.y
    graphs_noChangedDiameters.append(data_aux)
    for s in range(steps):
        # Find the pair of nodes with the maximum shortest path
        max_distance, node_pair=obtainDiameterNodes(G1)
        # Add an edge between the pair of nodes with the longest shortest path
        diameters.append(max_distance)
        
        if s==0:
            initial_diam=max_distance
        G1.add_edge(*node_pair)
        data_aux=from_networkx(G1)
        data_aux.x=data.x
        data_aux.y=data.y
        graphs_changedDiameters.append(data_aux)
        #draw_graph(from_networkx(G1))
        
        #Add edge to G2 without changing diameter
        non_edges = list(nx.non_edges(G2))
        random.shuffle(non_edges)
        for edge in non_edges:
            u,v=edge
            G_aux=G2.copy()
            G_aux.add_edge(u,v)
            max_distance,_=obtainDiameterNodes(G_aux)
            if max_distance==initial_diam:
                G2.add_edge(u,v)
                data_aux=from_networkx(G2)
                data_aux.x=data.x
                data_aux.y=data.y
                graphs_noChangedDiameters.append(data_aux)
                #draw_graph(from_networkx(G2))
                break    
    #Obtain diameter last step         
    max_distance,_=obtainDiameterNodes(G1)
    diameters.append(max_distance)    
                        
                               
    return graphs_changedDiameters,diameters,graphs_noChangedDiameters
            
def changeASPL(data,steps,edgesPerStep):    
    G1=to_networkx(data, to_undirected=True)
    G2=G1.copy()
    aspl_graphs_changed=[]
    aspl_graphs_noChanged=[]
    graphs_changed=[]
    graphs_noChanged=[]
    aspl_graphs_changed.append(nx.average_shortest_path_length(G1))
    aspl_graphs_noChanged.append(nx.average_shortest_path_length(G2))
    data_aux=from_networkx(G1)
    data_aux.x=data.x
    data_aux.y=data.y
    graphs_changed.append(data_aux)
    data_aux=from_networkx(G2)
    data_aux.x=data.x
    data_aux.y=data.y
    graphs_noChanged.append(data_aux)
    for s in range(steps):
        listEdges_g1=list(nx.non_edges(G1))
        random.shuffle(listEdges_g1)
        G1.add_edges_from(listEdges_g1[:edgesPerStep])
        aspl_graphs_changed.append(nx.average_shortest_path_length(G1))
        data_aux=from_networkx(G1)
        data_aux.x=data.x
        data_aux.y=data.y
        graphs_changed.append(data_aux)
        
        numAddedEdges = 0
        while numAddedEdges< edgesPerStep:
            u = random.choice(list(G2.nodes()))
            neighbors_u = list(G2.neighbors(u))
            if len(neighbors_u) > 0:
                v = random.choice(neighbors_u)
                # Get the neighbors of v (excluding u, to prevent selecting u again)
                neighbors_v = list(G2.neighbors(v))
                neighbors_v = [w for w in neighbors_v if w != u]                
                if len(neighbors_v) > 0:
                    w = random.choice(neighbors_v)
                    if not G2.has_edge(u, w):
                        G2.add_edge(u, w)
                        numAddedEdges+=1
        data_aux=from_networkx(G2)
        data_aux.x=data.x
        data_aux.y=data.y
        graphs_noChanged.append(data_aux)
        aspl_graphs_noChanged.append(nx.average_shortest_path_length(G2))
    return graphs_changed,aspl_graphs_changed,graphs_noChanged,aspl_graphs_noChanged