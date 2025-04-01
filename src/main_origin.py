# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    # config_dict = {
    #     'gpu_id': 0,
    #     'valid_metric': 'Recall@10',
    #     'use_gpu': False,
    #     'data_path': './data/',
    #     'dataset': 'sports',
    #     'inter_file_name': 'sports.inter',
    #     'USER_ID_FIELD': 'userID',
    #     'ITEM_ID_FIELD': 'itemID',
    #     'inter_splitting_label': 'x_label',
    #     'field_separator': '\s+',
    #     'read_csv_engine': 'python',
    #     'seed': [42, 2021, 2022],
    #     'NEG_PREFIX': 'neg_'
    # }
    config_dict = {
        'data_path': './data/',
        'use_gpu': False,
        'epochs' : 10,
        'metrics' : ["Recall"]
    }



    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


