import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils as ut
from dataloader import BidirectionalOneShotIterator, TrainDataset
from model import KGEModel


def main(args):
    args.cuda = torch.cuda.is_available()

    os.makedirs(args.save_path, exist_ok=True)

    ut.logger(args)
    tb_sw = SummaryWriter(log_dir=args.log_dir)
    wandb.init(project='ghkg', sync_tensorboard=True)

    e2id = ut.index('entities.dict', args)
    r2id = ut.index('relations.dict', args)
    args.nentity = len(e2id)
    args.nrelation = len(r2id)

    for k, v in sorted(vars(args).items()):
        logging.info(f'{k} = {v}')

    tr_q = ut.read(os.path.join(args.dataset, 'train.txt'), e2id, r2id, args.static)
    vd_q = ut.read(os.path.join(args.dataset, 'valid.txt'), e2id, r2id, args.static)
    ts_q = ut.read(os.path.join(args.dataset, 'test.txt'), e2id, r2id, args.static)
    logging.info(f'# Train = {len(tr_q)}')
    logging.info(f'# Valid = {len(vd_q)}')
    logging.info(f'# Test = {len(ts_q)}')

    al_q = tr_q + vd_q + ts_q

    tp_ix, tp_rix = ut.type_index(args) if args.negative_type_sampling or args.type_evaluation else (None, None)
    e_ix, u_ix = ut.users_index(args) if args.heuristic_evaluation else (None, None)

    mdl = nn.DataParallel(KGEModel(tp_ix, tp_rix, e_ix, u_ix, args))
    if args.cuda:
        mdl = mdl.cuda()

    logging.info('Model Parameter Configuration:')
    for name, param in mdl.named_parameters():
        if '_nn' not in name:
            logging.info(f'Parameter {name}: {param.size()}, require_grad = {param.requires_grad}')

    ev_ix = ut.event_index(tr_q)

    if args.do_train:
        tr_dl_s = DataLoader(TrainDataset(tr_q, tp_ix, tp_rix, ev_ix, 's', args),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=max(1, os.cpu_count() // 2))

        tr_dl_o = DataLoader(TrainDataset(tr_q, tp_ix, tp_rix, ev_ix, 'o', args),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=max(1, os.cpu_count() // 2))

        tr_it = BidirectionalOneShotIterator(tr_dl_s, tr_dl_o)

        lr = args.learning_rate
        wd = args.weight_decay
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=lr, weight_decay=wd)
        opt_sc = MultiStepLR(opt, milestones=list(map(int, args.learning_rate_steps.split(','))))

    if args.checkpoint != '':
        logging.info(f'Loading checkpoint {args.checkpoint} ...')
        chk = torch.load(os.path.join(args.checkpoint, 'checkpoint.chk'))
        init_stp = chk['step']
        mdl.load_state_dict(chk['mdl_state_dict'])
        if args.do_train:
            lr = chk['opt_state_dict']['param_groups'][0]['lr']
            opt.load_state_dict(chk['opt_state_dict'])
            opt_sc.load_state_dict(chk['opt_sc_state_dict'])
    else:
        logging.info('Randomly Initializing ...')
        init_stp = 1

    stp = init_stp

    logging.info('Start Training ...')
    logging.info(f'init_stp = {init_stp}')

    if args.do_train:
        logging.info(f'learning_rate = {lr}')

        logs = []
        bst_mtrs = {}
        for stp in range(init_stp, args.max_steps + 1):
            log = mdl.module.train_step(mdl, opt, opt_sc, tr_it, args)
            logs.append(log)

            if stp % args.log_steps == 0:
                mtrs = {}
                for mtr in logs[0].keys():
                    mtrs[mtr] = sum([log[mtr] for log in logs]) / len(logs)
                ut.log('Training average', stp, mtrs)
                logs.clear()
            ut.tensorboard_scalars(tb_sw, 'train', stp, log)

            if args.do_valid and stp % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset ...')
                mtrs = mdl.module.test_step(mdl, vd_q, al_q, ev_ix, args)
                if bst_mtrs.get(args.metric, None) is None or mtrs[args.metric] > bst_mtrs[args.metric]:
                    bst_mtrs = mtrs.copy()
                    var_ls = {'step': stp}
                    ut.save(mdl, opt, opt_sc, var_ls, args)
                ut.log('Valid', stp, mtrs)
                ut.tensorboard_scalars(tb_sw, 'valid', stp, mtrs)

        ut.tensorboard_hparam(tb_sw, bst_mtrs, args)

    if args.do_eval:
        logging.info('Evaluating on Training Dataset ...')
        mtrs = mdl.module.test_step(mdl, tr_q, al_q, ev_ix, args)
        ut.log('Test', stp, mtrs)
        ut.tensorboard_scalars(tb_sw, 'eval', stp, mtrs)

    if args.do_test:
        args.valid_approximation = 0
        args.test_log_steps = 100
        logging.info('Evaluating on Test Dataset ...')
        mdl.load_state_dict(torch.load(os.path.join(args.save_path, f'checkpoint.chk'))['mdl_state_dict'])
        mtrs = mdl.module.test_step(mdl, ts_q, al_q, ev_ix, args)
        ut.log('Test', stp, mtrs)
        ut.tensorboard_scalars(tb_sw, 'test', stp, mtrs)

    tb_sw.flush()
    tb_sw.close()


if __name__ == '__main__':
    main(ut.args())
