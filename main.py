import argparse
from loader import load_data
from train import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def print_setting(args):
    assert args.use_context or args.use_path
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('feature_type: ' + args.feature_type)

    print('use relational context: ' + str(args.use_context))
    if args.use_context:
        print('context_hops: ' + str(args.context_hops))
        print('neighbor_samples: ' + str(args.neighbor_samples))
        print('neighbor_agg: ' + args.neighbor_agg)

    print('use relational path: ' + str(args.use_path))
    if args.use_path:
        print('max_path_len: ' + str(args.max_path_len))
        print('path_type: ' + args.path_type)
        if args.path_type == 'rnn':
            print('path_samples: ' + str(args.path_samples))
            print('path_agg: ' + args.path_agg)
    print('=============================================')
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu') #, action='store_true')

    parser.add_argument('--dataset', type=str, default='ddb', help='dataset name')
    parser.add_argument('--epoch', type=int, default=4, help='number of epochs') # 20
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=4, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=10, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=5, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=4, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    #'''

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    # print(data)
    # train(args, data)


if __name__ == '__main__':
    main()
