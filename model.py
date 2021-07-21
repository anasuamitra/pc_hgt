import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator

class PathCon(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, params_for_paths):
        super(PathCon, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, params_for_paths)
        self._build_model()

    def _parse_args(self, args, n_relations, params_for_neighbors, params_for_paths):
        self.n_relations = n_relations
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type

        self.use_context = args.use_context
        if self.use_context:
            self.entity2edges = torch.LongTensor(params_for_neighbors[0]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[0])
            self.edge2entities = torch.LongTensor(params_for_neighbors[1]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[1])
            self.edge2relation = torch.LongTensor(params_for_neighbors[2]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[2])
            self.neighbor_samples = args.neighbor_samples
            self.context_hops = args.context_hops
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                self.neighbor_agg = CrossAggregator

        self.use_path = args.use_path
        if self.use_path:
            self.path_type = args.path_type
            if self.path_type == 'embedding':
                self.n_paths = params_for_paths[0] # 101
            elif self.path_type == 'rnn':
                self.max_path_len = args.max_path_len
                self.path_samples = args.path_samples
                self.path_agg = args.path_agg
                self.id2path = torch.LongTensor(params_for_paths[0]).cuda() if args.cuda \
                        else torch.LongTensor(params_for_paths[0])
                self.id2length = torch.LongTensor(params_for_paths[1]).cuda() if args.cuda \
                        else torch.LongTensor(params_for_paths[1])

    def _build_model(self):
        # define initial relation features
        if self.use_context or (self.use_path and self.path_type == 'rnn'):
            self._build_relation_feature()
        # print(self.relation_features.shape)
        # print(self.relation_features)
        '''torch.Size([15, 14])
        tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        '''
        self.scores = 0.0

        if self.use_context:
            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())  # define aggregators for each layer
        # print(self.aggregators)
        '''ModuleList(
          (0): CrossAggregator(
            (layer): Linear(in_features=210, out_features=64, bias=True)
          )
          (1): CrossAggregator(
            (layer): Linear(in_features=4160, out_features=64, bias=True)
          )
          (2): CrossAggregator(
            (layer): Linear(in_features=4160, out_features=64, bias=True)
          )
          (3): CrossAggregator(
            (layer): Linear(in_features=4096, out_features=14, bias=True)
          )
        )
        '''
        if self.use_path:
            if self.path_type == 'embedding':
                self.layer = nn.Linear(self.n_paths, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)

            elif self.path_type == 'rnn':
                self.rnn = nn.LSTM(input_size=self.relation_dim, hidden_size=self.hidden_dim, batch_first=True)
                self.layer = nn.Linear(self.hidden_dim, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch):
        if self.use_context:
            self.entity_pairs = batch['entity_pairs']
            self.train_edges = batch['train_edges']

        if self.use_path:
            if self.path_type == 'embedding':
                self.path_features = batch['path_features']
            elif self.path_type == 'rnn':
                self.path_ids = batch['path_ids']

        self.labels = batch['labels']

        self._call_model()

    def _call_model(self):
        self.scores = 0.

        if self.use_context:
            edge_list, mask_list = self._get_neighbors_and_masks(self.labels, self.entity_pairs, self.train_edges)
            self.aggregated_neighbors = self._aggregate_neighbors(edge_list, mask_list)
            self.scores += self.aggregated_neighbors

        if self.use_path:
            if self.path_type == 'embedding':
                self.scores += self.layer(self.path_features)

            elif self.path_type == 'rnn':
                rnn_output = self._rnn(self.path_ids)
                self.scores += self._aggregate_paths(rnn_output)

        self.scores_normalized = F.sigmoid(self.scores)

    def _build_relation_feature(self):
        if self.feature_type == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = torch.eye(self.n_relations).cuda() if self.use_gpu \
                    else torch.eye(self.n_relations)
        elif self.feature_type == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = torch.tensor(bow).cuda() if self.use_gpu \
                    else torch.tensor(bow)
        elif self.feature_type == 'bert':
            bert = np.load('../data/' + self.dataset + '/' + self.feature_type + '.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = torch.tensor(bert).cuda() if self.use_gpu \
                    else torch.tensor(bert)

        # the feature of the last relation (the null relation) is a zero vector
        self.relation_features = torch.cat([self.relation_features, 
                        torch.zeros([1, self.relation_dim]).cuda() if self.use_gpu \
                            else torch.zeros([1, self.relation_dim])], dim=0)

    def _get_neighbors_and_masks(self, relations, entity_pairs, train_edges):
        edges_list = [relations]
        # print('edges_list', edges_list[0].shape) # torch.Size([128])
        '''edges_list [tensor([ 2,  8,  2,  7,  0,  2,  2,  2,  2,  2,  2,  8,  2,  0,  0,  2,  6,  1,
             2,  8,  2,  8,  2,  2,  2,  2,  6,  2,  2,  2,  2,  2,  3,  0,  0,  8,
             2,  2, 12,  2,  2,  3,  2,  0,  2,  2,  8,  2,  0,  2,  2,  2,  0,  0,
             0,  2, 10,  2,  2,  2,  2,  2,  0,  2,  2,  0,  2,  2,  2,  2,  3,  0,
             2,  0,  2,  2,  0,  0,  0,  2,  8,  2,  2,  7,  2,  4,  2,  0,  2,  2,
             0,  0,  2,  2,  2,  7,  2,  2,  2,  2, 12,  2,  2,  2,  4,  0,  2,  8,
             2,  2,  2,  2,  2,  0,  2,  2,  2,  2,  2,  4,  2,  2,  2,  0,  2,  2,
             0,  0])]
        '''
        # print('train_edges', train_edges.shape) # torch.Size([128])
        ''' train_edges tensor([31416, 27836, 17681, 11669, 21801, 29433, 42554, 14638, 17312, 38745,
             2770,  1378,  8098, 24292, 37374,  1876,   744, 34917, 12007, 37565,
            10785, 32944, 10696, 41434, 19176,  3809, 31961, 34571, 14819,  1422,
            12163, 39952, 16218, 21560, 25730, 38611,  5080, 29540, 44037, 29583,
            12709, 39376, 13696, 23495,  6502, 41588, 32693, 12094, 26952, 19432,
            13937, 43515, 27295, 38016, 22646, 28596, 11246, 41526, 32929, 17843,
             1208, 40356, 33694, 15919, 11232, 23943, 18387, 32931,  5083, 28825,
             8404, 27073,  7033, 38118, 17539,  8779, 20933, 42864, 24533, 37033,
            36595, 17137, 37715, 31704, 41086,   636,  6152, 22990, 19514, 39213,
            20846, 30330, 17232, 42644, 43654,  6447, 35749, 14874,  7562, 27790,
            28480,  5353,  8080,  2649,  9441, 22833,  4528, 35831,  1739, 17214,
            20427, 11965, 34483, 23011, 40492, 15976, 19543, 18704,  9950,  2655,
            27813,   396, 20320, 27697, 13744,  3490, 42324, 20870])
        '''
        masks = []
        train_edges = torch.unsqueeze(train_edges, -1)  # [batch_size, 1]
        # print('train_edges', train_edges.shape) # torch.Size([128, 1])
        ''' train_edges tensor([[31416],
            [27836],
            [17681],
            [11669],
            [21801],
            [29433],
            [42554],
            [14638],
            [17312],
            [38745],
            [ 2770],
            [ 1378],
            [ 8098], ...]'''
        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs
            else:
                print('edges_list', edges_list[-1].view(-1))
                ''' edges_list tensor([33933, 41302, 35890,  ..., 21011, 42671, 44561]) '''
                neighbor_entities = torch.index_select(self.edge2entities, 0, 
                            edges_list[-1].view(-1)).view([self.batch_size, -1])
                print('neighbor_entities', neighbor_entities.shape)
                ''' neighbor_entities tensor([[2103, 8610, 2103,  ..., 6823, 9203, 9203],
                    [9203, 9203, 9203,  ..., 8816, 9203, 9203],
                    [ 813,  756,  816,  ..., 6179, 9203, 9203],
                    ...,
                    [ 463, 5492, 1353,  ..., 2695, 9203, 9203],
                    [8780, 8330, 4443,  ..., 8330, 9203, 9203],
                    [ 307,  950,  307,  ..., 6594, 9203, 9203]])
                '''
            neighbor_edges = torch.index_select(self.entity2edges, 0, 
                            neighbor_entities.view(-1)).view([self.batch_size, -1])
            # print('neighbor_edges', neighbor_edges.shape) # torch.Size([128, 20])
            # print('neighbor_edges', neighbor_edges)
            ''' neighbor_edges tensor([[ 1857, 12911, 12926,  ...,  2152,  2134,  2135],
                [40880, 23207, 24239,  ..., 15772, 15712, 44561],
                [ 2829,   449,  7852,  ..., 41163, 27612, 44561],
                ...,
                [10406,   614,  9621,  ..., 42903, 42571, 44561],
                [22765,  2710, 41925,  ..., 31523, 31859, 31862],
                [23809,  9337, 20141,  ..., 19659, 19665, 44561]])
            '''
            edges_list.append(neighbor_edges)

            mask = neighbor_edges - train_edges  # [batch_size, -1]
            print(mask)
            mask = (mask != 0).float()
            masks.append(mask)

        return edges_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators

        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [torch.index_select(self.relation_features, 0, edge_list[0])]
        for edges in edge_list[1:]:
            relations = torch.index_select(self.edge2relation, 0, 
                            edges.view(-1)).view(list(edges.shape)+[-1])
            edge_vectors.append(torch.index_select(self.relation_features, 0, 
                            relations.view(-1)).view(list(relations.shape)+[-1]))

        # shape of edge vectors:
        # [[batch_size, relation_dim],
        #  [batch_size, 2 * neighbor_samples, relation_dim],
        #  [batch_size, (2 * neighbor_samples) ^ 2, relation_dim],
        #  ...]

        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            neighbors_shape = [self.batch_size, -1, 2, self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size, -1, 2, self.neighbor_samples, 1]

            for hop in range(self.context_hops - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=edge_vectors[hop + 1].view(neighbors_shape),
                                    masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = edge_vectors[0].view([self.batch_size, self.n_relations])
        return res

    def _rnn(self, path_ids):
        path_ids = path_ids.view([self.batch_size * self.path_samples])  # [batch_size * path_samples]
        paths = torch.index_select(self.id2path, 0, 
                path_ids.view(-1)).view(list(path_ids.shape)+[-1])  # [batch_size * path_samples, max_path_len]
        # [batch_size * path_samples, max_path_len, relation_dim]
        path_features = torch.index_select(self.relation_features, 0, 
                paths.view(-1)).view(list(paths.shape)+[-1])
        lengths = torch.index_select(self.id2length, 0, path_ids)  # [batch_size * path_samples]

        output, _ = self.rnn(path_features)
        output = torch.cat([torch.zeros(output.shape[0], 1, output.shape[2]).cuda() if self.use_gpu \
                    else torch.zeros(output.shape[0], 1, output.shape[2]), output], dim=1)
        output = output.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(output.shape[0], 1, output.shape[-1]))

        output = self.layer(output)
        output = output.view([self.batch_size, self.path_samples, self.n_relations])

        return output

    def _aggregate_paths(self, inputs):
        # input shape: [batch_size, path_samples, n_relations]

        if self.path_agg == 'mean':
            output = torch.mean(inputs, dim=1)  # [batch_size, n_relations]
        elif self.path_agg == 'att':
            assert self.use_context
            aggregated_neighbors = self.aggregated_neighbors.unsqueeze(1)  # [batch_size, 1, n_relations]
            attention_weights = torch.sum(aggregated_neighbors * inputs, dim=-1)  # [batch_size, path_samples]
            attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, path_samples]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, path_samples, 1]
            output = torch.sum(attention_weights * inputs, dim=1)  # [batch_size, n_relations]
        else:
            raise ValueError('unknown path_agg')

        return output

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores, model.labels))
        loss.backward()
        optimizer.step()

        return loss.item()
    
    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
        return acc, model.scores_normalized.tolist()