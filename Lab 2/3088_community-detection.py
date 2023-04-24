from traceback import print_tb
import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os
import itertools

# NOTE: STUDENT_AM replaced with AM_3088 because functions cannot start with a number

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])
        
    else: # REQUIRED_NUM_COLORS > len(my_color_list)   
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]
 
    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions):
    plt.figure(figsize=(10,10))
    
    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G, 
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency 
        width = 0.5                 # edge-width
        )

    # changed this so code execution continues without having to close the plot, so multiple graphs can be showed
    plt.show(block=False)

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN 
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)
        
        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')
        
        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: #...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            
                else:
                    # LOAD GRAPH FROM DATA SET...
                    G,node_names_list = \
                    AM_3088_read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            
            else: # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO...

                else: # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2: 
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else: # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []         # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True
            
        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples,community_tuples):

    breakWhileLoop = False

    while not breakWhileLoop:
            print(bcolors.OKGREEN 
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])
            
                    # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC) 
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else: 
                        G = \
                        AM_3088_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE...

                    G = \
                    AM_3088_add_hamilton_cycle_to_graph(G,node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G...
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else: # ...len(my_option_list) == 2... 
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         # ...ERASE previous node positions...

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                
                # flag for the percentage of nodes for the betweenness calculation in the gervin-newman functions
                node_percent = -1
                
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        graph_layout = 'spring'     # DEFAULT graph layout == spring
                
                    else: # ...NUM_OPTIONS == 3...
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])

                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:
                        
                        # setting the flag for the betweenness calculation node percentage
                        try:
                            node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
                        except ValueError:
                            node_percent = -1

                        while node_percent>1 or node_percent<=0:
                            print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
                            try:
                                node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
                            except ValueError:
                                node_percent = -1

                        _,community_tuples = AM_3088_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions,node_percent=node_percent)

                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:

                        # setting the flag for the betweenness calculation node percentage
                        try:
                            node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
                        except ValueError:
                            node_percent = -1

                        while node_percent>1 or node_percent<=0:
                            print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
                            try:
                                node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
                            except ValueError:
                                node_percent = -1

                        _,community_tuples = AM_3088_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions,node_percent=node_percent)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                # different min number for comparison when graph starts with 1 connected component (used in experiment 11)
                # MAX_NUM_DIVISIONS = min( 40*NUM_COMPONENTS , np.floor(NUM_NODES/4) )


                # alternative number of division up to number of nodes
                # MAX_NUM_DIVISIONS = np.floor(NUM_NODES)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for 
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        
                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: #...NUM_OPTIONS == 3...
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS...
                    if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components." + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProvide a number of divisions in { ",NUM_COMPONENTS,",",MAX_NUM_DIVISIONS,"}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        hierarchy_of_community_tuples, community_tuples = AM_3088_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)

            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                max_modularity_partition = AM_3088_determine_opt_community_structure(G,hierarchy_of_community_tuples)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: # ...NUM_OPTIONS == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        node_positions = AM_3088_visualize_communities(G,community_tuples,graph_layout,node_positions)

            elif my_option_list[0] == 'E':
                #EXIT the program execution...
                quit()

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ### 

########################################################################################
############################### MY ROUTINES LIBRARY ENDS ############################### 
########################################################################################

########################################################################################
########################## AM_3088 ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################
def AM_3088_read_graph_from_csv(NUM_LINKS):

    # read csv - the file must be in the same folder/directory as this code
    fb_links = pd.read_csv('fb-pages-food.edges')

    # get the first NUM_LINKS
    fb_links_df = fb_links.head(NUM_LINKS).copy()

    # remove self loop edges
    fb_links_loopless_df = fb_links_df[fb_links_df['node_1'] != fb_links_df['node_2']].copy()

    # construct the graph
    G = nx.from_pandas_edgelist(fb_links_loopless_df, "node_1", "node_2", create_using=nx.Graph())

    # get the nodes in the graph
    node_names_list = list(G.nodes())

    return G, node_names_list


######################################################################################################################
# ...(a) AM_3088 IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def AM_3088_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions,node_percent=1):
    # print(  bcolors.ENDC 
    #         + "\tCalling routine " 
    #         + bcolors.HEADER + "AM_3088_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)\n" 
    #         + bcolors.ENDC)

    start_time = time.time()

    community_tuples_before = [tuple(c) for c in sorted(nx.connected_components(G),key=len, reverse=True)]

    # largest connected component in the graph
    largest_cc = max(nx.connected_components(G), key=len)

    # create a subgraph out of the largest connected component
    largest_cc_subgraph = G.subgraph(largest_cc).copy()

    # calculate the number of nodes that the betweenness calculation will use
    node_number = round(node_percent*len(largest_cc_subgraph.nodes()))

    # calculate the betweenness of the subgraph
    betweenness = cntr.edge_betweenness_centrality(largest_cc_subgraph,k=node_number) 

    # sort
    betweenness_sorted = {k: v for k, v in sorted(betweenness.items(), key=lambda item: item[1],reverse=True)}

    # keep removing edges in descending order of the betweenness until the graph is not connected anymore
    for key,value in betweenness_sorted.items():
        largest_cc_subgraph.remove_edge(key[0],key[1])
        if not nx.is_connected(largest_cc_subgraph):
            break
    
    # find the two components that make the subgraph
    subgraph_components = [tuple(c) for c in nx.connected_components(largest_cc_subgraph)]

    # remove the first component and add the two new ones in the initial graph
    community_tuples_before.remove(tuple(largest_cc))
    combined_components = subgraph_components + community_tuples_before

    # sort by most nodes first etc
    community_tuples_sorted = [tuple(sorted(c)) for c in sorted(combined_components,key=len, reverse=True)]
   
    end_time = time.time()
   
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    print(bcolors.ENDC  + "\tCalculation of betweenness by using",node_percent*100,"% of",largest_cc_subgraph.number_of_nodes(),"nodes","\n")

    # return the community that was removed and communities after the split
    return tuple(sorted(largest_cc)),community_tuples_sorted


######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def AM_3088_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions,node_percent=1):
    # print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "AM_3088_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    start_time = time.time()

    initial_comms = [tuple(sorted(c)) for c in sorted(nx.connected_components(G),key=len, reverse=True)]

    # largest connected component *EXPECTED* to be splitted with girvan_newman
    largest_cc = max(nx.connected_components(G), key=len)

    # callback function for girvan-newman
    def most_central_edge(Gr):
        # calculate the number of nodes that the betweenness calculation will use
        node_number = round(node_percent*len(Gr.nodes()))
        # calculate the centrality
        centrality = cntr.edge_betweenness_centrality(Gr,k=node_number)
        return max(centrality,key=centrality.get)

    # calling networkx girvan-newman
    split_communities = comm.girvan_newman(G,most_valuable_edge=most_central_edge)

    # components that are created after splitting of the largest connected component
    community_tuples = [tuple(sorted(c)) for c in next(split_communities)]

    # find two new components (the ones added will not be in the initial communities before the split)
    new_components = [i for i in community_tuples if i not in initial_comms]
    
    # SOS /!\ found out the largest component sometimes is not used for some reason and another component gets split, so calculate the component that got split here
    largest_cc_calculated_from_split = new_components[0] + new_components[1]
    largest_cc_calculated_from_split = sorted(largest_cc_calculated_from_split)

    # print if girvan-newman chose different connected component other than the largest one
    if tuple(largest_cc_calculated_from_split) != tuple(sorted(largest_cc)):
        # print('largest_cc:',tuple(sorted(largest_cc)))
        # print('cc used in networkx girvan-newman:',tuple(largest_cc_calculated_from_split))
        print(bcolors.WARNING  + "\tNetworkx chose different component other than the largest one!" +  bcolors.ENDC)

    # most nodes first etc
    community_tuples_sorted = [tuple(c) for c in sorted(community_tuples,key=len, reverse=True)]

    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    print(bcolors.ENDC  + "\tCalculation of betweenness by using",node_percent*100,"% of",G.subgraph(largest_cc).number_of_nodes(),"nodes","\n")

    # return the community that was removed and communities after the split
    return tuple(sorted(largest_cc_calculated_from_split)), community_tuples_sorted


######################################################################################################################
def AM_3088_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions):
    # print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "AM_3088_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    start_time = time.time()

    # select algorithm to be used in the computation of the binary hierarchy below
    algorithm = -1
    try:
        algorithm = int(input('\tProvide the algorithm for the split 1:own, 2:nx : '))
    except ValueError:
        algorithm = -1

    while algorithm not in [1,2]:
        print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
        try:
            algorithm = int(input('\tProvide the algorithm for the split 1:own, 2:nx : '))
        except ValueError:
            algorithm = -1

    # select the percentage of nodes for the calculation of the betweenness 
    node_percent = -1
    try:
        node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
    except ValueError:
        node_percent = -1

    while node_percent>1 or node_percent<=0:
        print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
        try:
            node_percent = round(float(input('\tProvide the percentage of node samples to be used for the betweenness estimation (0-1]: ')),2)
        except ValueError:
            node_percent = -1

    # initial K communities
    initial_comms = [tuple(c) for c in sorted(nx.connected_components(G),key=len, reverse=True)]

    # initial K partitions
    partitions = len(initial_comms)
    
    hierarchy = []

    tempG = G.copy()

    while partitions < number_of_divisions:

        # calculate current communities before girvan newman
        before_comms = [tuple(sorted(c)) for c in sorted(nx.connected_components(tempG),key=len, reverse=True)]

        if algorithm == 2: # nx
            lc,com_tuples = AM_3088_use_nx_girvan_newman_for_communities(G=tempG,graph_layout=graph_layout,node_positions=node_positions,node_percent=node_percent)
        else: # own
            lc,com_tuples = AM_3088_one_shot_girvan_newman_for_communities(G=tempG,graph_layout=graph_layout,node_positions=node_positions,node_percent=node_percent)
        

        # find which two communities were added
        added_comms = [c for c in com_tuples if c not in before_comms]

        # LCC, LC1, LC2
        hierarchy.append([lc,added_comms[0],added_comms[1]])

        # update graph with returned communities (removing the LC and replacing it with two components LC1,LC2)
        subgraphs = []

        # for each community make a subgraph
        for node_list in com_tuples:
            subgraphs.append(tempG.subgraph(node_list).copy())

        # combine the subgraphs
        tempG = nx.compose_all(subgraphs)

        partitions+=1
        print('Current partition:',partitions)

    print('Total partitions:',partitions)

    # calculation to return for visualization
    final_communities = [tuple(c) for c in nx.connected_components(tempG)]

    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
                        + "using the BUILT-IN girvan-newman algorithm, for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

    
    return hierarchy, final_communities


######################################################################################################################
def AM_3088_determine_opt_community_structure(G,hierarchy_of_community_tuples):
    # print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "AM_3088_determine_opt_community_structure(G,hierarchy_of_community_tuples)" + bcolors.ENDC +"\n")

    # sort each tuple in the hierarchy
    for i in range(len(hierarchy_of_community_tuples)):
        for j in range(3):
            hierarchy_of_community_tuples[i][j] = tuple(sorted(hierarchy_of_community_tuples[i][j]))

    # sorted by increasing node id
    current_community_tuples = [tuple(sorted(c)) for c in sorted(nx.connected_components(G),key=len, reverse=True)]

    # K initial communities
    initial_comms = len(current_community_tuples)
    
    modularity_values = []

    # add initial modularity
    modu = comm.modularity(G,current_community_tuples)
    modularity_values.append(modu)

    newG = G.copy()
    
    max_mod_G = newG.copy()
    max_mod = abs(modu)
    max_mod_communities = []

    # at each iteration remove the largest community and add the new ones as specified by the hierarchy
    for partition in hierarchy_of_community_tuples:
        # at the current partition of the hierarchy remove the LC and add the LC1,LC2
        current_community_tuples.remove(partition[0])
        current_community_tuples.append(partition[1])
        current_community_tuples.append(partition[2])

        # current partition modularity
        modu = comm.modularity(G,current_community_tuples)
        modularity_values.append(modu)

        # rebuild the graph by constructing all the subgraphs of the new communities and then joining them
        subgraphs = []
        for node_list in current_community_tuples:
            subgraphs.append(newG.subgraph(node_list).copy())
        newG = nx.compose_all(subgraphs)

        if abs(modu) > max_mod:
            max_mod_G = newG.copy()
            max_mod = abs(modu)
            max_mod_communities = current_community_tuples


    # calculate the maximum modularity
    abs_mod_values = [abs(v) for v in modularity_values]
    max_mod = max(abs_mod_values)

    # initial communities is not part of the hierarchy
    if  modularity_values.index(max_mod) == 0:
        print('\tInitial partition',initial_comms,'with the max modularity',max_mod,':')

    # print the hierarchy that has the optimal modularity
    else:
        print('\tPartition',modularity_values.index(max_mod)+1+initial_comms-1,'with the max modularity',max_mod,':')
        print(hierarchy_of_community_tuples[modularity_values.index(max_mod)-1])
    
    graph_layout = 'spring'
    node_positions = []
    AM_3088_visualize_communities(max_mod_G,max_mod_communities,graph_layout,node_positions)

    # select the range of partitions to plot
    # start point
    partsstart = -1
    try:
        print('\tPartition range:',initial_comms,'-',initial_comms+len(modularity_values)-1)
        partsstart = int(input('\tProvide the start of the partitions to plot: '))
    except ValueError:
        partsstart = -1

    while partsstart >initial_comms+len(modularity_values)-1 or partsstart<0:
        print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
        try:
            print('\tPartition range:',initial_comms,'-',initial_comms+len(modularity_values)-1)
            partsstart = int(input('\tProvide the start of the partitions to plot: '))
        except ValueError:
            partsstart = -1

    # end point
    partsend = -1
    try:
        print('\tPartition range:',partsstart,'-',len(modularity_values)+initial_comms-1)
        partsend = int(input('\tProvide the end of the partitions to plot: '))
    except ValueError:
        partsend = -1

    while partsend >len(modularity_values)+initial_comms-1 or partsend<partsstart:
        print(bcolors.WARNING + "\tInvalid input. Try again..." + bcolors.ENDC)
        try:
            print('\tPartition range:',partsstart,'-',len(modularity_values)+initial_comms-1)
            partsend = int(input('\tProvide the end of the partitions to plot: '))
        except ValueError:
            partsend = -1

    partsstart -= initial_comms
    partsend -= initial_comms

    modularity_values_show = modularity_values[partsstart:partsend+1]

    # give the maximum modularity red color for easy identification in the plot
    colors = []
    for modu in modularity_values_show:
        if modu==max_mod:
            colors.append('r')
        else:
            colors.append('b')

    # bar plot of the modularity values in the selected range
    plt.bar(range(initial_comms,initial_comms+len(modularity_values_show)),height=modularity_values_show,color=colors)

    title = 'Modularity starting with '+str(partsstart+initial_comms)+' communities and reaching '+str(partsend+initial_comms)+' communities'
    plt.title(title)

    ticks_labels = [p+initial_comms for p in range(partsstart,partsend+1)]
    plt.xticks(range(initial_comms,initial_comms+len(modularity_values_show)),labels=ticks_labels)
    
    # code execution continues without having to close the plot, so multiple graphs can be showed
    plt.show(block=False)

    # when the initial K communities has the largest modularity
    if  modularity_values.index(max_mod) == 0:
        return [(),(),()]
    # normally return the respective hierarchy of the max modularity
    return hierarchy_of_community_tuples[modularity_values.index(max_mod)-1]


######################################################################################################################
def AM_3088_add_hamilton_cycle_to_graph(G,node_names_list):
    # print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "AM_3088_add_hamilton_cycle_to_graph(G,node_names_list)" + bcolors.ENDC +"\n")

    # NOTE: there is the option to check if there is an (in)direct path between two nodes, instead of just a direct edge, by using nx.has_path instead of has_edge

    tempG = G.copy()

    for ni in range(len(node_names_list)-1):
        # if not nx.has_path(tempG,node_names_list[ni], node_names_list[ni+1]):
        if not tempG.has_edge(node_names_list[ni], node_names_list[ni+1]):
            tempG.add_edge(node_names_list[ni], node_names_list[ni+1])

    # edge between first and last nodes 
    # if not nx.has_path(tempG,node_names_list[0], node_names_list[-1]):
    if not tempG.has_edge(node_names_list[0], node_names_list[-1]):
        tempG.add_edge(node_names_list[0], node_names_list[-1])

    return tempG
    

######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
def AM_3088_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY):
    # print(  bcolors.ENDC     + "\tCalling routine " 
    #         + bcolors.HEADER + "AM_3088_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)" 
    #         + bcolors.ENDC   + "\n")

    tempG = G.copy()

    for node in tempG.nodes():
        # find the nodes that aren't neighbors with the current node
        non_neigh = [n for n in nx.non_neighbors(tempG,node)]

        # for the NUM_RANDOM_EDGES per node
        for i in range(NUM_RANDOM_EDGES):

            # choose a random non-neighboring node
            dest_node = random.choice(non_neigh)

            # with EDGE_ADDITION_PROBABILITY add an adge
            if (random.random() < EDGE_ADDITION_PROBABILITY):
                tempG.add_edge(node,dest_node)

    return tempG


######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def AM_3088_visualize_communities(G,community_tuples,graph_layout,node_positions):
    # print(bcolors.ENDC      + "\tCalling routine " 
    #                         + bcolors.HEADER + "AM_3088_visualize_communities(G,community_tuples,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    if not community_tuples:
        print(bcolors.WARNING  + "\tCommunities not created yet (C or D menu options), cannot show graph." +  bcolors.ENDC)
        return
    # change graph with input community_tuples
    tempG = nx.Graph()
    subgraphs = []
    for node_list in community_tuples:
        subgraphs.append(G.subgraph(node_list).copy())
    tempG = nx.compose_all(subgraphs)

    # assign colors to each community
    random_colors = my_random_color_list_generator(len(community_tuples))
    color_map = []
    for node in tempG:
        for com_index in range(len(community_tuples)):
            if node in community_tuples[com_index]:
                color_map.append(random_colors[com_index])
    
    # node_positions = [] # uncomment this to reset the positions og the communities
    node_positions = my_graph_plot_routine(tempG,color_map,'blue','solid',graph_layout,node_positions)

    return(node_positions)


########################################################################################
########################### AM_3088 ROUTINES LIBRARY ENDS ###########################
########################################################################################


########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################
G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G,node_names_list,node_positions = my_menu_graph_construction(G,node_names_list,node_positions)

my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples,community_tuples=community_tuples)