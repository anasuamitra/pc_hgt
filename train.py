import argparse, collections, os, random
from os.path import join
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from peach.help import *
from peach.common import load_list_from_file, load_tsv, file_exists, dir_exists, save_json, load_json

from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer

from loader import load_data
from pchgt.kb_dataset import KbDataset
from pchgt.models import BertForPairScoring, RobertaForPairScoring
from pchgt.graph_bert import GraphBertConfig
from pchgt.metric import calculate_metrics_for_link_prediction, safe_ranking
from pchgt.utils_fn import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str, help="model class, one of [bert, roberta]")
    parser.add_argument("--dataset", type=str, default="ddb")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--neg_weights", default=None, type=str)
    parser.add_argument("--distance_metric", default="euclidean", type=str)   # 默认距离度量为mlp
    parser.add_argument("--hinge_loss_margin", default=1., type=float)
    parser.add_argument("--pos_weight", default=1, type=float)
    parser.add_argument("--loss_weight", default=1, type=float)
    parser.add_argument("--cls_loss_weight", default=1, type=float)
    parser.add_argument("--cls_method", default="cls", type=str)

    # extra parameters for prediction
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--collect_prediction", action="store_true")
    parser.add_argument("--prediction_part", default="0,1", type=str)

    # parameter for negative sampling
    parser.add_argument("--type_cons_neg_sample", action="store_true")
    parser.add_argument("--type_cons_ratio", default=0, type=float)
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu') #, action='store_true')
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

    print("Parsing args")
    define_hparams_training(parser)
    args = parser.parse_args()
    setup_prerequisite(args)

    data = load_data(args) #edges, paths, len(edge_type2index), neighbor_params, path_params
    # print(data[0]) #[[(2458, 633, 12), (2260, 6810, 0)]], edges = [train_edges, val_edges, test_edges]
    # print(data[1])
    ''' [<44561x101 sparse matrix of type '<class 'numpy.float64'>'
        with 2136 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
        with 234 stored elements in List of Lists format>, <4000x101 sparse matrix of type '<class 'numpy.float64'>'
        with 234 stored elements in List of Lists format>] '''
    # print(data[2]) # 14
    # print(data[3])
    '''[array([[    0,  2915,  2916, ..., 11495, 16778, 11114],
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
    print(data[4])
    residual_type = 'graph_raw'
    k = 7
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = 2
    graph_size = 1
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        model_class = BertForPairScoring
    elif args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForPairScoring
    elif args.model_class == "graph_bert":
        config_class = GraphBertConfig(residual_type = residual_type, k=k, x_size=100, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)

        model_class = RobertaForPairScoring

if __name__ == '__main__':
    main()