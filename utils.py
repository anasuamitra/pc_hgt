import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from collections import defaultdict
from littleballoffur import SnowBallSampler, RandomWalkSampler

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


# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
        # print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    print('pid %d  done' % pid)
    return ht2paths


def bfs(head, tails, e2re, max_path_len):
    # put length-1 paths into all_paths
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]
    #print(head) #5251
    #print(tails) #{8273, 4555, 390}
    #print(all_paths) #[[(0, 8273)], [(2, 390)], [(2, 4555)]]
    #print()
    # p = 0
    # for length in range(2, max_path_len + 1):
    #     while p < len(all_paths) and len(all_paths[p]) < length:
    #         path = all_paths[p]
    #         #print(path) #[(2, 4026), (2, 1498)]
    #         last_entity_in_path = path[-1][1]
    #         #print(last_entity_in_path) #1498
    #         entities_in_path = set([head] + [i[1] for i in path])
    #         #print(entities_in_path) #{1498, 4026, 4027}
    #         for edge in e2re[last_entity_in_path]:
    #             # append (relation, entity) to the path if the new entity does not appear in this path before
    #             if edge[1] not in entities_in_path:
    #                 all_paths.append(path + [edge])
    #             #print("Inside for: ", edge, path + [edge])
    #             '''
    #             Inside for:  (4, 947) [(2, 4026), (2, 1498), (4, 947)]
    #             Inside for:  (2, 6066) [(2, 4026), (2, 1498), (2, 6066)]
    #             Inside for:  (4, 946) [(2, 4026), (2, 1498), (4, 946)]
    #             Inside for:  (2, 2485) [(2, 4026), (2, 1498), (2, 2485)]
    #             Inside for:  (2, 6388) [(2, 4026), (2, 1498), (2, 6388)]
    #             Inside for:  (2, 5857) [(2, 4026), (2, 1498), (2, 5857)]
    #             Inside for:  (4, 3416) [(2, 4026), (2, 1498), (4, 3416)]
    #             Inside for:  (2, 6456) [(2, 4026), (2, 1498), (2, 6456)]
    #             Inside for:  (4, 801) [(2, 4026), (2, 1498), (4, 801)]
    #             Inside for:  (4, 8414) [(2, 4026), (2, 1498), (4, 8414)]
    #             Inside for:  (2, 4026) [(2, 4026), (2, 1498), (2, 4026)]
    #             '''
    #         p += 1

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            # print(path) #[(2, 4026), (2, 1498)]
            last_entity_in_path = path[-1][1]
            p += 1

    ht2paths = defaultdict(set)
    for path in all_paths:
        tail = path[-1][1]
        if tail in tails:  # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path]))
    #print(ht2paths)
    '''
    defaultdict(<class 'set'>, {(8350, 2047): {(2,), (2, 0, 0)}, 
    (8350, 8712): {(2,), (2, 2, 2), (2, 0, 0), (2, 2)}, 
    (8350, 3893): {(0, 0, 2), (2,), (2, 0, 0), (2, 2)}, 
    (8350, 5876): {(2,), (0, 0), (2, 2, 2), (2, 2)}, 
    (8350, 7756): {(2,), (2, 0, 0)}, 
    (8350, 8322): {(2, 0), (0,), (2, 2, 0)}})
    '''
    print()
    return ht2paths


def count_paths(triplets, ht2paths, train_set):
    res = []

    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)]
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])})
        res.append([list(i) for i in path_list])
        ''' [[2], [2, 2]], [], [], [], [[2], [2, 6]], [], [], [], [], [], [], [], [] '''
    return res


def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths #{(3, 2, 3): 0, (4, 2): 1, (4,): 2, (2, 4): 3, (4, 4): 4}
                id2length.append(len(path)) #[3, 2, 1, 2, 2, 1, 2, 3]
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                ''' [[3, 2, 3], [4, 2, 14], [4, 14, 14], [2, 4, 14], [4, 4, 14], [2, 14, 14], [2, 2, 14], [11, 11, 11] '''
                n_paths += 1
    return path2id, id2path, id2length


def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)

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


def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape
