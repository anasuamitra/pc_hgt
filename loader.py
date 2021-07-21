import os
import re
import pickle as pkl
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
import multiprocessing as mp
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from littleballoffur import SnowBallSampler, RandomWalkSampler
from tqdm import tqdm


entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
edge2entities = []  # each row in edge2entities is the two entities connected by this edge
edge2relation = []  # each row in edge2relation is the relation type of this edge
e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity

G = nx.empty_graph()
S, HH = list(), dict()
node_mapping, not_in_g = list(), list()
bfs_subgraphs, bfs_contexts, bfs_e2nodes, bfs_e2relations, bfs_e2edges = dict(), dict(), dict(), dict(), dict()
dfs_subgraphs, dfs_contexts, dfs_e2nodes, dfs_e2relations, dfs_e2edges = dict(), dict(), dict(), dict(), dict()
null_entity, null_relation, null_edge = 0, 0, 0

def build_kg(train_data):
    global entity2edge_set, entity2edges, edge2entities, edge2relation, e2re, null_entity, null_relation, null_edge

    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet
        if args.use_context:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)
        if args.use_path:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details
    if args.use_context or args.use_path:
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)
        for i in range(len(node_id2index) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}
            # sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=args.neighbor_samples,
            #                                      replace=len(entity2edge_set[i]) < args.neighbor_samples)
            # entity2edges.append(sampled_neighbors)

def get_h2t(train_triplets, valid_triplets, test_triplets):
    head2tails = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        head2tails[head].add(tail)
    return head2tails

def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num
    #print(n_cores) #8
    #print(pool) #<multiprocessing.pool.Pool object at 0x7f5e28a90bd0>
    #print(range_list) #[[0, 1143], [1143, 2286], [2286, 3429], [3429, 4572], [4572, 5715], [5715, 6858], [6858, 8001], [8001, 9144]]
    return n_cores, pool, range_list

def dfs(head, tails, e2re, max_path_len):
    global G, S, HH, node_mapping, dfs_subgraphs, dfs_contexts, dfs_e2nodes, dfs_e2relations, dfs_e2edges, not_in_g
    ht2paths = defaultdict(set); flag = False
    if args.use_path:
        if head not in dfs_subgraphs: dfs_subgraphs[head] = list()
        for item in S:
            if head in list(item.nodes()):
                idx = S.index(item)
                tmp = node_mapping[idx]
                rel_head = tmp[head]
                H = HH[idx]
                flag = True
        if not flag:
            # print(head)
            not_in_g.append(head)
            return ht2paths
        # H = nx.relabel_nodes(item, tmp)
        ''' Node path sampling '''
        if H.degree()[rel_head] == 0 or (H.degree()[rel_head] == 2 or rel_head in list(nx.nodes_with_selfloops(H))):
            g = nx.Graph()
            g.add_node(null_entity, name="NULL", id=null_entity)
            g.add_edge(null_entity, null_entity, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
            dfs_subgraphs[head].append(g)
        for samples in range(args.path_samples):
            for length in range(2, max_path_len + 1):
                if H.degree()[rel_head] == 0 or (H.degree()[rel_head] == 2 or rel_head in list(nx.nodes_with_selfloops(H))):
                    continue
                else:
                    try:
                        model = RandomWalkSampler(number_of_nodes=length)
                        nG = model.sample(H, rel_head)
                    except:
                        if min(H.degree()[rel_head])>1:
                            model = RandomWalkSampler(number_of_nodes=min(H.degree()[rel_head], length))
                            nG = model.sample(H, rel_head)
                        else:
                            continue
                        inv_map = {v: k for k, v in tmp.items()}
                        rev_nG = nx.relabel_nodes(nG, inv_map)
                        dfs_subgraphs[head].append(rev_nG)

        for path_graphs in dfs_subgraphs[head]:
            item_list = list()
            for n in list(path_graphs.nodes()):
                item_list.append(n)
            dfs_contexts[head] = item_list
            dfs_e2nodes[head] = item_list
            rel_list = list()
            edge_list = list()
            for e in list(path_graphs.edges(data=True)):
                rel_list.append(e[2]['relation'])
                edge_list.append(e[2]['id'])
            dfs_e2relations[head] = rel_list
            dfs_e2edges[head] = edge_list
            tail = item_list[-1]
            if tail in tails:  # if this path ends at tail
                ht2paths[(head, tail)].add(tuple([i for i in rel_list]))

        # print(ht2paths)
        '''
        defaultdict(<class 'set'>, {(8350, 2047): {(2,), (2, 0, 0)}, 
        (8350, 8712): {(2,), (2, 2, 2), (2, 0, 0), (2, 2)}, 
        (8350, 3893): {(0, 0, 2), (2,), (2, 0, 0), (2, 2)}, 
        (8350, 5876): {(2,), (0, 0), (2, 2, 2), (2, 2)}, 
        (8350, 7756): {(2,), (2, 0, 0)}, 
        (8350, 8322): {(2, 0), (0,), (2, 2, 0)}})
        '''
        # print()

    return ht2paths

# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(dfs(head, tails, e2re, max_path_len))
    #     # print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    # print('pid %d  done' % pid)
    return ht2paths

def count_all_paths_with_mp(e2re, max_path_len, head2tails):
    n_cores, pool, range_list = get_params_for_mp(len(head2tails))
    results = pool.map(count_all_paths, zip([e2re] * n_cores,
                                            [max_path_len] * n_cores,
                                            [head2tails[i[0]:i[1]] for i in range_list],
                                            range(n_cores)))
    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)

    return res

def count_paths(triplets, ht2paths, train_set):
    res = []

    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)] # Extract all the paths with (h,t)
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])})
        res.append([list(i) for i in path_list])

    return res

def get_paths(train_triplets, valid_triplets, test_triplets):
    directory = 'dataset/' + args.dataset + '/cache/'
    length = str(args.max_path_len)

    if not os.path.exists(directory):
        os.mkdir(directory)

    if os.path.exists(directory + 'train_dfs_' + length + '.pkl'):
        print('loading paths from files ...')
        train_paths = pkl.load(open(directory + 'train_dfs_' + length + '.pkl', 'rb'))
        valid_paths = pkl.load(open(directory + 'val_dfs_' + length + '.pkl', 'rb'))
        test_paths = pkl.load(open(directory + 'test_dfs_' + length + '.pkl', 'rb'))

    else:
        print('counting paths from head to tail ...')
        head2tails = get_h2t(train_triplets, valid_triplets, test_triplets)
        ht2paths = count_all_paths_with_mp(e2re, args.max_path_len, [(k, v) for k, v in head2tails.items()])
        # print(ht2paths)
        train_set = set(train_triplets)
        train_paths = count_paths(train_triplets, ht2paths, train_set)
        valid_paths = count_paths(valid_triplets, ht2paths, train_set)
        test_paths = count_paths(test_triplets, ht2paths, train_set)
        # print(train_paths)
        # print(len(train_paths), len(valid_paths), len(test_paths)) # 89122 8000 8000: for context=3
        print('dumping paths to files ...')
        pkl.dump(train_paths, open(directory + 'train_dfs_' + length + '.pkl', 'wb'))
        pkl.dump(valid_paths, open(directory + 'val_dfs_' + length + '.pkl', 'wb'))
        pkl.dump(test_paths, open(directory + 'test_dfs_' + length + '.pkl', 'wb'))

    # if using rnn and no path is found for the triplet, put an empty path into paths
    if args.path_type == 'rnn':
        for paths in train_paths + valid_paths + test_paths:
            if len(paths) == 0:
                paths.append([])

    return train_paths, valid_paths, test_paths

def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                n_paths += 1
    return path2id, id2path, id2length

def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features

def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)
    # print(res)
    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]

def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths]
            sampled_path_ids_for_triplet = np.random.choice(
                path_ids_for_triplet, size=path_samples, replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)
    return res

def load_data(model_args):
    global args, node_id2index, edge_type2index, G, S, HH, node_mapping, bfs_subgraphs, bfs_contexts, not_in_g, \
    bfs_e2nodes, bfs_e2relations, bfs_e2edges, dfs_e2nodes, dfs_e2relations, dfs_e2edges, null_entity, null_relation, null_edge
    args = model_args

    ''' Loading networked data from pickle file. '''
    network = pkl.load(open('dataset/{}.pkl'.format(args.dataset), "rb"))
    node_id2index = network["node_id2index"]
    edge_type2index = network["edge_type2index"]
    train_edges = network["splits"]['train_edges']
    val_edges = network["splits"]['val_edges']
    test_edges = network["splits"]['test_edges']
    ''' Making the network undirected. '''
    # train_edges = train_edges + ([(tup[1], tup[0], tup[2]) for tup in train_edges])
    # val_edges = val_edges + ([(tup[1], tup[0], tup[2]) for tup in val_edges])
    # test_edges = test_edges + ([(tup[1], tup[0], tup[2]) for tup in test_edges])
    print('Reading entities and relations ...')
    print('Reading train, validation, and test edges ...')
    print('Processing the heterogeneous graph ...')

    # print(len(entity2edges)) #9204
    ''' Creating a graph for sampling contexts and paths. '''
    edges = [train_edges, val_edges, test_edges]
    all_edges = train_edges
    all_edges += val_edges
    all_edges += test_edges
    null_entity = len(node_id2index)
    null_relation = len(edge_type2index)
    null_edge = len(all_edges)
    # print(len(train_edges), len(val_edges), len(test_edges)) # 44561 4000 4000
    ''' Storing train-edge related statistics. '''
    build_kg(train_edges)
    ''' Build train & whole graph. '''
    df = pd.DataFrame(all_edges, columns=['target', 'source', 'relation']) # input edge format is: h-t-r
    G = nx.from_pandas_edgelist(df, 'target', 'source', ['relation'])
    df1 = pd.DataFrame(train_edges, columns=['target', 'source', 'relation'])  # input edge format is: h-t-r
    G1 = nx.from_pandas_edgelist(df1, 'target', 'source', ['relation'])
    # print(nx.number_of_isolates(G)) #147 The below print statements are for train data
    # comp_list = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # print(comp_list) #DDB: [9053, 2, 2] Rest are 147 singleton nodes
    for edge_idx, triplet in enumerate(all_edges): # all_edges related attributes are stored.
        head_idx, tail_idx, relation_idx = triplet
        G[head_idx][tail_idx]['weight'] = 1.0
        G[head_idx][tail_idx]['id'] = edge_idx
        #if 'id' not in G[head_idx]: G[head_idx]['id'] = head_idx
        #if 'id' not in G[tail_idx]: G[tail_idx]['id'] = tail_idx
        G[head_idx][tail_idx]['relation'] = relation_idx
        G[head_idx][tail_idx]['name'] = "N_NULL"
    for edge_idx, triplet in enumerate(train_edges): # train_edges related attributes are stored.
        head_idx, tail_idx, relation_idx = triplet
        G1[head_idx][tail_idx]['weight'] = 1.0
        G1[head_idx][tail_idx]['id'] = edge_idx
        #if 'id' not in G1[head_idx]: G1[head_idx]['id'] = head_idx
        #if 'id' not in G1[tail_idx]: G1[tail_idx]['id'] = tail_idx
        G1[head_idx][tail_idx]['relation'] = relation_idx
        G1[head_idx][tail_idx]['name'] = "N_NULL"
    for i in range(len(node_id2index) + 1):
        if i in list(nx.isolates(G)): #nx.is_isolate(G, i):
            G.add_edge(i, i, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
        elif i not in G.nodes(): # i in list(nx.isolates(G)):
            G.add_node(i, name="NULL", id=null_entity)
            G.add_edge(i, i, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
        if i in list(nx.isolates(G1)): #nx.is_isolate(G, i):
            G1.add_edge(i, i, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
        elif i not in G1.nodes(): # i in list(nx.isolates(G)):
            G1.add_node(i, name="NULL", id=null_entity)
            G1.add_edge(i, i, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
    ''' Sampling for all connected components of the underlying graph. '''
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)] #all subgraphs
    idx = 0
    for item in S:
        tmp = {}
        for i, I in enumerate(list(item.nodes())): tmp[I] = i
        node_mapping.append(tmp) #absolute->relative node id mapping
        HH[idx] = nx.relabel_nodes(item, tmp, copy=True) #mapped subgraphs
        idx += 1
    S1, HH1 = list(), dict()
    node_mapping1 = list()
    S1 = [G1.subgraph(c).copy() for c in nx.connected_components(G1)]
    idx = 0
    for item in S1:
        tmp = {}
        for i, I in enumerate(list(item.nodes())): tmp[I] = i
        node_mapping1.append(tmp)
        HH1[idx] = nx.relabel_nodes(item, tmp, copy=True)
        idx += 1
    ''' Data-structures to store path-related information. '''
    bfs_e2nodes = {new_list: [] for new_list in range(len(list(G.nodes())))}
    bfs_e2relations = {new_list: [] for new_list in range(len(list(G.nodes())))}
    bfs_e2edges = {new_list: [] for new_list in range(len(list(G.nodes())))}
    dfs_e2nodes = {new_list: [] for new_list in range(len(list(G.nodes())))}
    dfs_e2relations = {new_list: [] for new_list in range(len(list(G.nodes())))}
    dfs_e2edges = {new_list: [] for new_list in range(len(list(G.nodes())))}
    ''' Checking for pre-existing node contexts. '''
    directory = 'dataset/' + args.dataset + '/cache/'
    length = str(args.neighbor_samples)
    if args.use_context:
        if os.path.exists(directory + 'train_bfs_' + length + '.pkl'):
            print('Loading contexts from files ...')
            bfs_e2edges = pkl.load(open(directory + 'train_bfs_' + length + '.pkl', 'rb'))
        else:
            ''' Node context sampling. '''
            for j, n in enumerate(tqdm(G1.nodes())):
                if n not in bfs_subgraphs: bfs_subgraphs[n] = list()
                for item in S1:
                    if n in list(item.nodes()):
                        idx = S1.index(item)
                        tmp = node_mapping1[idx]
                        rel_head = tmp[n]
                        H = HH1[idx]
                if H.degree()[rel_head] == 0 or (
                        H.degree()[rel_head] == 2 or rel_head in list(nx.nodes_with_selfloops(H))):
                    g = nx.Graph()
                    g.add_node(null_entity, name="NULL", id=null_entity)
                    g.add_edge(null_entity, null_entity, name="NULL", id=null_edge, relation=null_relation, weight=1.0)
                    bfs_subgraphs[n].append(g)
                else:
                    try:
                        model = SnowBallSampler(number_of_nodes=args.neighbor_samples)
                        nG = model.sample(H, rel_head)
                    except:
                        model = SnowBallSampler(number_of_nodes=min(H.degree()[rel_head], args.neighbor_samples))
                        nG = model.sample(H, rel_head)
                    inv_map = {v: k for k, v in tmp.items()}
                    rev_nG = nx.relabel_nodes(nG, inv_map)
                    bfs_subgraphs[n].append(rev_nG)

                for path_graphs in bfs_subgraphs[n]:
                    item_list = list()
                    for m in list(path_graphs.nodes()):
                        item_list.append(m)
                    bfs_contexts[n] = item_list
                    bfs_e2nodes[n] = item_list
                    rel_list = list()
                    edge_list = list()
                    for e in list(path_graphs.edges(data=True)):
                        rel_list.append(e[2]['relation'])
                        edge_list.append(e[2]['id'])
                    bfs_e2relations[n] = rel_list
                    bfs_e2edges[n] = edge_list

            print('Dumping paths to files ...')
            pkl.dump(bfs_e2edges, open(directory + 'train_bfs_' + length + '.pkl', 'wb'))
            # entity2edges = bfs_e2edges

        for key, value in bfs_e2edges.items():
            # t_list = []
            # print(value)
            # for item in value:
            #     t_list.append(int(item))
            # entity2edges.append(np.array(t_list).astype(int))
            # print()
            # entity2edges.append(t_list)
            if len(value) < args.neighbor_samples:
                value += [null_edge] * (args.neighbor_samples - len(value))
            elif len(value) > args.neighbor_samples: # It can happen since we are sampling nodes not edges
                value = value[:args.neighbor_samples]
            # print(np.array(value).shape)
            entity2edges.append(np.array(value).astype(np.int32))
        neighbor_params = [np.array(entity2edges).reshape((len(bfs_e2edges.keys()), args.neighbor_samples)), np.array(edge2entities), np.array(edge2relation)]
    else:
        neighbor_params = None
    # print(neighbor_params[0])
    ''' {9201: [72724, 72734, 46927, 72726, 72725, 72723, 72728, 72722, 72729, 72727], 
    9202: [55904, 52017, 52019, 87461, 68982, 40029, 68310, 73046, 73044, 39972, 73045], 9203: [73122]} '''
    # print(neighbor_params)
    ''' [array([[    0,  2915,  2916, ..., 11495, 16778, 11114],
       [   28,    34,    35, ..., 42903, 42571, 44561],
       [11503,  1127, 11509, ...,  5945,  1116,  5948],
       ...,
       [36163, 36173, 10366, ..., 36161, 36168, 36166],
       [19343, 15456, 15458, ..., 36485, 36483,  3411],
       [44561, 44561, 44561, ..., 44561, 44561, 44561]], dtype=int32), array([[   0,    1],
       [   2,    1],
       [   4,    1],
       ...,
       [2458,  633],
       [2260, 6810],
       [9203, 9203]]), array([ 0,  0,  0, ..., 12,  0, 14])]
    '''

    if args.use_path:
        train_paths, valid_paths, test_paths = get_paths(train_edges, val_edges, test_edges)
        # print(not_in_g)
        print(train_paths)
        path2id, id2path, id2length = get_path_dict_and_length(train_paths, valid_paths, test_paths, len(edge_type2index), args.max_path_len)
        if args.path_type == 'embedding':
            print('transforming paths to one hot IDs ...')
            paths = one_hot_path_id(train_paths, valid_paths, test_paths, path2id)
            path_params = [len(path2id)]
        elif args.path_type == 'rnn':
            paths = sample_paths(train_paths, valid_paths, test_paths, path2id, args.path_samples)
            path_params = [id2path, id2length]
        else:
            raise ValueError('unknown path type')
    else:
        paths = [None] * 3
        path_params = None
    # print("path2id", path2id)
    '''path2id {(3, 2, 3): 0, (4, 2): 1, (2, 4): 2, (4, 4): 3, (2, 2): 4, (11, 11, 11): 5, (11, 2): 6, (2, 3): 7, (4, 12): 8, (12, 2): 9, (7, 8): 10, (6, 7): 11, (7, 7): 12, (8, 2): 13, (3, 2): 14, (2, 11): 15, (12, 12, 11): 16, (2, 2, 12): 17, (6, 6): 18, (7, 2): 19, (4, 4, 4): 20, (0, 11): 21, (3, 2, 2): 22, (5, 4): 23, (2, 2, 2): 24, (2, 12): 25, (6, 0): 26, (0, 8): 27, (12, 2, 2): 28, (0, 0, 12): 29, (12, 4): 30, (2, 11, 2): 31, (2, 0): 32, (11, 4): 33, (2, 12, 2): 34, (9, 8, 8): 35, (8, 8): 36, (2, 5): 37, (2, 2, 5): 38, (5, 0): 39, (3, 3): 40, (4, 6): 41, (2, 2, 3): 42, (2, 7): 43, (4, 5, 5): 44, (5, 5): 45, (2, 3, 4): 46, (4, 5): 47, (0, 2): 48, (11, 4, 4): 49, (6, 2): 50, (3, 0, 0): 51, (8, 12): 52, (2, 8): 53, (2,): 54, (4, 2, 2): 55, (12, 12): 56, (2, 2, 11): 57, (12, 8): 58, (2, 3, 3): 59, (5, 2): 60, (10, 2): 61, (2, 4, 2): 62, (2, 7, 2): 63, (7, 2, 2): 64, (5, 2, 2): 65, (11, 0, 0): 66, (0, 0): 67, (0, 11, 0): 68, (0, 12, 0): 69, (0, 2, 0): 70, (1, 0): 71, (0, 1): 72, (4, 0, 0): 73, (2, 0, 0): 74, (0, 4, 0): 75, (2, 10): 76, (11, 2, 2): 77, (10, 12): 78, (2, 5, 2): 79, (6, 4): 80, (8, 9): 81, (0, 9, 0): 82, (6, 8): 83, (4, 2, 4): 84, (12, 0): 85, (12, 8, 8): 86, (2, 6): 87, (12, 6): 88, (11, 11): 89, (10, 10): 90, (0, 4): 91, (9, 8): 92, (9, 9): 93, (2, 9): 94, (8, 12, 8): 95, (8, 9, 8): 96, (2, 4, 5): 97, (4, 8): 98, (1, 1): 99, (2, 2, 4): 100}
    '''
    # print("id2path", id2path)
    '''id2path [[3, 2, 3], [4, 2, 14], [2, 4, 14], [4, 4, 14], [2, 2, 14], [11, 11, 11], [11, 2, 14], [2, 3, 14], [4, 12, 14], [12, 2, 14], [7, 8, 14], [6, 7, 14], [7, 7, 14], [8, 2, 14], [3, 2, 14], [2, 11, 14], [12, 12, 11], [2, 2, 12], [6, 6, 14], [7, 2, 14], [4, 4, 4], [0, 11, 14], [3, 2, 2], [5, 4, 14], [2, 2, 2], [2, 12, 14], [6, 0, 14], [0, 8, 14], [12, 2, 2], [0, 0, 12], [12, 4, 14], [2, 11, 2], [2, 0, 14], [11, 4, 14], [2, 12, 2], [9, 8, 8], [8, 8, 14], [2, 5, 14], [2, 2, 5], [5, 0, 14], [3, 3, 14], [4, 6, 14], [2, 2, 3], [2, 7, 14], [4, 5, 5], [5, 5, 14], [2, 3, 4], [4, 5, 14], [0, 2, 14], [11, 4, 4], [6, 2, 14], [3, 0, 0], [8, 12, 14], [2, 8, 14], [2, 14, 14], [4, 2, 2], [12, 12, 14], [2, 2, 11], [12, 8, 14], [2, 3, 3], [5, 2, 14], [10, 2, 14], [2, 4, 2], [2, 7, 2], [7, 2, 2], [5, 2, 2], [11, 0, 0], [0, 0, 14], [0, 11, 0], [0, 12, 0], [0, 2, 0], [1, 0, 14], [0, 1, 14], [4, 0, 0], [2, 0, 0], [0, 4, 0], [2, 10, 14], [11, 2, 2], [10, 12, 14], [2, 5, 2], [6, 4, 14], [8, 9, 14], [0, 9, 0], [6, 8, 14], [4, 2, 4], [12, 0, 14], [12, 8, 8], [2, 6, 14], [12, 6, 14], [11, 11, 14], [10, 10, 14], [0, 4, 14], [9, 8, 14], [9, 9, 14], [2, 9, 14], [8, 12, 8], [8, 9, 8], [2, 4, 5], [4, 8, 14], [1, 1, 14], [2, 2, 4]]
    '''
    # print("id2length", id2length)
    '''id2length [3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 1, 3, 2, 3, 2, 3, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 3]
    '''
    # print(path_params) #[144], [101]
    print(paths)
    '''
        [<89122x144 sparse matrix of type '<class 'numpy.float64'>'
        with 7282 stored elements in List of Lists format>, <8000x144 sparse matrix of type '<class 'numpy.float64'>'
        with 792 stored elements in List of Lists format>, <8000x144 sparse matrix of type '<class 'numpy.float64'>'
        with 792 stored elements in List of Lists format>]
        
        [<44561x101 sparse matrix of type '<class 'numpy.float64'>'
        with 2136 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
        with 234 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
        with 234 stored elements in List of Lists format>]

    '''
    return edges, paths, len(edge_type2index), neighbor_params, path_params
