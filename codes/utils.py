import argparse
import itertools
import json
import logging
import os
import re
from collections import defaultdict

import pandas as pd
import torch


def args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--id', required=True, type=str)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--static', action='store_true')

    parser.add_argument('--static_dim', default=256, type=int)
    parser.add_argument('--absolute_dim', default=256, type=int)
    parser.add_argument('--relative_dim', default=256, type=int)

    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--gamma', default=6.0, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--lmbda', default=0.0, type=float)

    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--learning_rate_steps', default='100000', type=str)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--criterion', default='NS', type=str, choices=['CE', 'MR', 'NS'])

    parser.add_argument('--negative_sample_size', default=256, type=int)
    parser.add_argument('--negative_time_sample_size', default=8, type=int)
    parser.add_argument('--negative_adversarial_sampling', action='store_true')
    parser.add_argument('--negative_type_sampling', action='store_true')
    parser.add_argument('--negative_max_time_gap', default=1555200, type=int)

    parser.add_argument('--heuristic_evaluation', action='store_true')
    parser.add_argument('--type_evaluation', action='store_true')

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)

    parser.add_argument('--max_steps', default=200000, type=int)

    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--metric', default='MRR', type=str, choices=['H1', 'H3', 'H10', 'MR', 'MRR'])

    parser.add_argument('--mode', default='both', type=str, choices=['head', 'tail', 'both', 'time', 'full'])

    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--valid_approximation', default=0, type=int)
    parser.add_argument('--log_steps', default=1000, type=int)
    parser.add_argument('--test_log_steps', default=1000, type=int)
    parser.add_argument('--log_dir', default=None, type=str)

    parser.add_argument('--timezone', default='America/Montreal', type=str)

    return parser.parse_args()


def event_index(tr_q):
    tr_ix = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in tr_q:
        tr_ix[s][r].append(t)
        tr_ix[o][r].append(t)

    ix = defaultdict(lambda: defaultdict(list))
    for k in tr_ix:
        for r in tr_ix[k]:
            ix[k][r] = sorted(set(tr_ix[k][r]))

    return ix


def index(fn, args):
    with open(os.path.join(args.dataset, fn)) as f:
        ix = dict()
        for l in f:
            i, r = l.strip().split('\t')
            ix[r] = int(i)
    return ix


def read(fn, e2id, r2id, stt=False):
    q = []
    with open(fn) as f:
        for l in f:
            s, r, o, t = l.strip().split('\t')
            q.append((e2id[s], r2id[r], e2id[o], 0 if stt else int(t)))
    return q


def type_index(args):
    with open(os.path.join(args.dataset, 'entities.dict'), 'r') as f:
        e_ix = dict(map(lambda x: x.split(), f.read().split('\n')[:-1]))

    ix, tr_ix = dict(), dict()
    regx = re.compile(r'^/(\w+)/.*$')
    for i, e in e_ix.items():
        tp = re.findall(regx, e)[0]
        if tp not in ix:
            ix[tp] = list()
        ix[tp].append(int(i))
        tr_ix[int(i)] = tp
    return ix, tr_ix


def log(md, stp, mtrs):
    for mtr, val in mtrs.items():
        logging.info(f'{md} {mtr} at step {stp}: {val}')


def logger(args):
    log_file = os.path.join(args.save_path or args.checkpoint, 'train.log' if args.do_train else 'test.log')

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')])


def save(mdl, opt, opt_sc, var_ls, args):
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    torch.save({
        **var_ls,
        'mdl_state_dict': mdl.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'opt_sc_state_dict': opt_sc.state_dict(),
    }, os.path.join(args.save_path, f'checkpoint.chk'))


def tensorboard_scalars(tb_sw, md, stp, mtrs):
    for mtr, val in mtrs.items():
        tb_sw.add_scalar(f'{md}/{mtr}', val, stp)


def tensorboard_hparam(tb_sw, mtrs, args):
    hparams_exc = [
        'do_train', 'do_valid', 'do_test', 'do_eval',
        'test_batch_size'
        'valid_steps', 'log_steps', 'test_log_steps',
    ]
    hparams_dict = {hparam: getattr(args, hparam) for hparam in vars(args) if hparam not in hparams_exc}
    tb_sw.add_hparams(hparams_dict, mtrs)


def users_index(args):
    with open(os.path.join(args.dataset, 'entities.dict'), 'r') as f:
        re_ix = dict(map(lambda x: x.split()[::-1], f.read().split('\n')[:-1]))

    tr = pd.read_csv(os.path.join(args.dataset, 'train.txt'), sep='\t', names=['s', 'r', 'o', 't'])

    r_e = tr[tr['o'].str.startswith('/repo/')][['o', 's']]
    r_e = r_e.rename(columns={'o': 'repo', 's': 'entity'})

    e_u = tr[tr['s'].str.startswith('/user/')].groupby('o')['s'].apply(list)
    e_u = e_u.reset_index(name='users').rename(columns={'o': 'entity'})

    r_u = r_e.merge(e_u, on='entity', how='left')[['repo', 'users']]
    r_u['users'] = r_u.users.apply(lambda x: x if type(x) == list else [])
    r_u = r_u.groupby('repo')['users'].apply(lambda x: list(itertools.chain.from_iterable(x)))
    r_u = r_u.reset_index(name='users')
    r_u['users'] = r_u.users.apply(lambda x: [int(re_ix[y]) for y in x])

    vd = pd.read_csv(os.path.join(args.dataset, 'valid.txt'), sep='\t', names=['s', 'r', 'o', 't'])
    ts = pd.read_csv(os.path.join(args.dataset, 'test.txt'), sep='\t', names=['s', 'r', 'o', 't'])
    al = pd.concat([tr, vd, ts])

    i_r = al[al['o'].str.startswith('/repo/') & al['s'].str.startswith('/issue/')]
    i_r = i_r.groupby('s')['o'].apply(lambda x: list(x)[0]).reset_index(name='repo')
    i_r = i_r.rename(columns={'s': 'issue'})
    i_r['issue'] = i_r.issue.apply(lambda x: int(re_ix[x]))

    p_r = al[al['o'].str.startswith('/repo/') & al['s'].str.startswith('/pr/')]
    p_r = p_r.groupby('s')['o'].apply(lambda x: list(x)[0]).reset_index(name='repo')
    p_r = p_r.rename(columns={'s': 'pr'})
    p_r['pr'] = p_r.pr.apply(lambda x: int(re_ix[x]))

    r_u_ix = r_u.set_index('repo').to_dict()['users']
    i_r_ix = i_r.set_index('issue').to_dict()['repo']
    p_r_ix = p_r.set_index('pr').to_dict()['repo']

    return {**i_r_ix, **p_r_ix}, r_u_ix
