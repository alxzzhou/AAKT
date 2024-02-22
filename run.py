import os
from argparse import ArgumentParser

import aakt_entry

prefix_dict = {
    'ednet': ['sequence5000', 'tags_ednet', 'EdNet-5000', ],
    'a09': ['assistment2009', 'tags_assistment09', 'ASSISTMENT-09', ],
    'a09f': ['assistment2009_full', 'tags_assistment09', 'ASSISTMENT-09-FULL', ],
    'a17': ['assistment2017', 'tags_assistment17', 'ASSISTMENT-17', ],
    'junyi': ['junyi', 'tags_junyi', 'junyi', ],
    'jf': ['junyi_full', 'tags_junyi', 'junyi-FULL', ]
}

arg_dict = {
    '--max_seq_len': 500,
    '--n_layer': 4,
    '--n_embd': 128,
    '--n_head': 8,
    '--epochs': 10,
    '--learning_rate': 1e-3,
    '--global_batch_size': 64,
    '--mini_batch_size': 64,
    '--eval': 1,
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ednet')
    args = parser.parse_args()
    model_dataset = args.dataset

    assert model_dataset in prefix_dict.keys()

    model_args = arg_dict
    fp, tp, rp = prefix_dict[model_dataset][:3]

    cmd = f'python {aakt_entry.__file__}'
    cmd += f' --file_prefix {fp} --tag_prefix {tp} --record_prefix {rp}'
    for k, v in model_args.items():
        if callable(v):
            cmd += f' {k} {v(model_dataset)}'
        else:
            cmd += f' {k} {v}'

    os.system(cmd)
