import torch
import numpy as np
from collections import defaultdict
from model import PathCon
from utils import sparse_to_tuple


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, paths, n_relations, neighbor_params, path_params = data
    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32)) # train edge indices
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))
    # print(test_entity_pairs)
    ''' tensor([[ 438,  431],
        [  86, 6100],
        [1167, 2256],
        ...,
        [4902, 8662],
        [6638, 4732],
        [ 437, 4673]])
    '''
    # print(train_edges) # tensor([    0,     1,     2,  ..., 36558, 36559, 36560])

    train_paths, valid_paths, test_paths = paths
    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))
    # print(test_labels) #tensor([2, 2, 2,  ..., 0, 2, 2])
    # print(test_labels.shape) #torch.Size([4000]) 8000
    # print(valid_labels.shape) #torch.Size([4000]) 8000
    # print(train_labels.shape) #torch.Size([36561]) 89122

    # define the model
    model = PathCon(args, n_relations, neighbor_params, path_params)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    for step in range(args.epoch):
        # shuffle training data
        index = np.arange(len(train_labels))
        # print(index) #[    0     1     2 ... 36558 36559 36560] Edge indexes, [    0     1     2 ... 44558 44559 44560]
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
            # print(train_entity_pairs)
            '''tensor([[1436, 6663],
                [2997, 2996],
                [4912, 1744],
                ...,
                [ 211, 4646],
                [ 682, 6134],
                [1643,  424]])
            '''
            # print(train_edges) #tensor([21488,  3809, 10265,  ..., 11588, 17198, 11441]) Shuffled train edge indices
        if args.use_path:
            train_paths = train_paths[index]
            # print(train_paths); print() # Encountered paths to metapath id association matrix.
            '''(35442, 9)	1.0
              (35442, 10)	1.0
              (35442, 11)	1.0
              (35442, 17)	1.0
              (35442, 18)	1.0
              (35442, 21)	1.0
              (35442, 29)	1.0
              (35442, 37)	1.0
              (35442, 89)	1.0
              (35442, 92)	1.0
              (35442, 800)	1.0
              (35442, 848)	1.0
              (35442, 938)	1.0
              (35442, 939)	1.0
              (35442, 940)	1.0
              (35442, 1116)	1.0
            '''
        train_labels = train_labels[index]
        # print(train_labels) #tensor([8, 2, 2,  ..., 0, 2, 2])
        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            s += args.batch_size
            # print(loss)
            ''' 2.6711084842681885
                2.6509976387023926
                2.579814910888672
                1.8282731771469116
                3.0045530796051025
                1.36283278465271
                1.883766770362854
            '''
        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('final results\n%s' % final_res)


def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    feed_dict = {}
    # print(start, end)
    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
            # print(train_edges[start:end])
            '''tensor([  931, 22551, 18989, 13317,   861,  9281, 28524, 23647, 11344, 11601,
                26268, 30327, 11013, 13825, 25138, 34799, 35280, 32748, 24645,  5254,
                18539, 17704,  6182, 33571, 18646, 14870,  5264,  5282, 34542, 18978,
                22286, 18278, 17317, 28428, 28530,  2040, 21260, 34172,  4742,   892,
                27702, 24364, 21034, 11363, 34560, 23409, 23924,   650,  9857,  3768,
                32661, 15279, 27489, 22452, 17544, 27819, 12851, 14751, 13202,  4970,
                 6904, 13532, 17758, 22478,  3482,  8671, 31922, 18032, 25244,  9177,
                12284,  9099, 27319,  4085, 31322, 17637, 33055, 20008, 21634,   211,
                14012, 18843,    28, 17453, 23866,  7504,  7410, 28706,  3731, 15401,
                10394, 25766, 13952, 11040, 30609,  7020, 11422, 28336, 30233,  3815,
                11284, 30511, 34114, 23648,  5192, 12642,  1395,  8536,  6211, 34767,
                25213, 10975, 31767, 29448, 36221, 29721, 32982, 23034,  8625, 29735,
                 8303, 36277,   897, 36261,  9800, 29848, 10216, 33033])
            '''
        else:
            # for evaluation no edges should be masked out
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                        else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    if args.use_path:
        if args.path_type == 'embedding':
            '''[<44561x101 sparse matrix of type '<class 'numpy.float64'>'
            with 2136 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
            with 234 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
            with 234 stored elements in List of Lists format>]'''
            indices, values, shape = sparse_to_tuple(paths[start:end])
            # print(indices)
            # print(values)
            # print(shape)
            '''[[ 10  36]
                 [ 19  67]
                 [ 22  67]
                 [ 38  67]
                 [ 60   4]
                 [ 62  15]
                 [ 91  14]
                 [115  67]]
                [1. 1. 1. 1. 1. 1. 1. 1.]
                (128, 101)
                ------------
                [[ 25  24]
                 [ 38   4]
                 [ 44   4]
                 [ 58   4]
                 [ 69  67]
                 [ 79  12]
                 [111   4]
                 [115   4]]
                [1. 1. 1. 1. 1. 1. 1. 1.]
                (128, 101)
            '''
            indices = torch.LongTensor(indices).cuda() if args.cuda else torch.LongTensor(indices)
            values = torch.Tensor(values).cuda() if args.cuda else torch.Tensor(values)
            feed_dict["path_features"] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(shape)).to_dense()
            # print(feed_dict["path_features"].shape) # torch.Size([128, 101])
        elif args.path_type == 'rnn':
            feed_dict["path_ids"] = torch.LongTensor(paths[start:end]).cuda() if args.cuda \
                    else torch.LongTensor(paths[start:end])

    feed_dict["labels"] = labels[start:end]
    # print(feed_dict)
    '''{'entity_pairs': tensor([[ 229, 6483],
        [  28,  117],
        [1595,  615],
        [ 203,  297],
        [4772, 3036],
        [ 773, 2297],
        [2208, 2194],
        [1048, 1358],
        [6652, 6645],..}, 'path_features': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]), 'labels': tensor([ 6,  2,  2,  7,  8,  5,  8,  2,  0,  2,  2,  2,  2,  2,  0,  0,  2,  4,
         2,  2,  2,  0,  2,  2,  0,  2,  2,  2, 10,  3,  4,  2,  2,  0,  2,  2,
         8,  2,  2,  2,  2,  2,  0,  7,  2,  2,  2,  2,  2,  2,  0,  2,  0,  0,
         0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  2,  2,  0,  2,
         8,  2,  2,  2,  4,  2,  2,  0,  0,  2,  8,  5,  0,  8,  2,  3,  2,  2,
         2, 12,  2,  0,  2,  0,  4,  8,  2,  0,  2,  2,  0,  2,  0,  4,  2,  0,
         2,  0,  2,  0,  2,  6,  2,  2,  2,  2,  2,  4,  2,  2,  7,  2,  2,  2,
         2,  8])}
    '''
    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5
