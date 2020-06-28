
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataloader import TestDataset


def train_step(mdl, opt, opt_sc, tr_it, args):
    mdl.train()

    opt.zero_grad()

    pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel, smpl_w, md = next(tr_it)
    smpl_w = smpl_w.squeeze(dim=1)
    if args.cuda:
        pos = pos.cuda()
        neg = neg.cuda()
        neg_abs = neg_abs.cuda()
        neg_rel = neg_rel.cuda()
        neg_abs_s_rel = neg_abs_s_rel.cuda()
        neg_abs_o_rel = neg_abs_o_rel.cuda()
        smpl_w = smpl_w.cuda()

    pos_sc = mdl(pos)
    neg_sc = mdl((pos, neg, neg_abs, neg_rel, neg_abs_s_rel, neg_abs_o_rel), md)
    if args.criterion == 'NS':
        pos_sc = F.logsigmoid(pos_sc).squeeze(dim=1)
        if args.negative_adversarial_sampling:
            neg_sc = (F.softmax(neg_sc * args.alpha, dim=1).detach() * F.logsigmoid(-neg_sc)).sum(dim=1)
        else:
            neg_sc = F.logsigmoid(-neg_sc).mean(dim=1)

        pos_lss = -(smpl_w * pos_sc).sum() / smpl_w.sum()
        neg_lss = -(smpl_w * neg_sc).sum() / smpl_w.sum()

        lss = (pos_lss + neg_lss) / 2
        lss_log = {'pos_loss': pos_lss.item(), 'neg_loss': neg_lss.item()}
    elif args.criterion == 'CE':
        trg = torch.zeros(pos_sc.size(0)).long()
        if args.cuda:
            trg = trg.cuda()
        lss = F.cross_entropy(torch.cat([pos_sc, neg_sc], dim=1), trg)
        lss_log = {}
    elif args.criterion == 'MR':
        trg = torch.ones(pos_sc.size(0)).long()
        if args.cuda:
            trg = trg.cuda()
        pos_sc = pos_sc.repeat(1, neg_sc.size(1)).view(-1, 1)
        lss = F.margin_ranking_loss(pos_sc, neg_sc.view(-1, 1), trg, margin=args.gamma)
        lss_log = {}

    reg_log = {}
    if args.lmbda != 0.0:
        reg = args.lmbda * (mdl.module.w_rp.norm(p=3) ** 3 +
                            mdl.module.w_e.norm(p=3) ** 3 +
                            mdl.module.e_emb.norm(p=3) ** 3 +
                            mdl.module.abs_d_amp_emb.norm(p=3) ** 3 +
                            mdl.module.abs_m_amp_emb.norm(p=3) ** 3)
        lss += reg
        reg_log = {'regularization': reg.item()}

    lss.backward()
    opt.step()
    opt_sc.step()

    return {**reg_log, **lss_log, 'loss': lss.item()}


def test_step(mdl, ts_q, al_q, ev_ix, tp_ix, tp_rix, e_ix, u_ix, args):
    mdl.eval()

    ts_dls = []
    if args.mode in ['head', 'both', 'full']:
        ts_dls.append((DataLoader(TestDataset(ts_q, al_q, ev_ix, 's', args),
                                  batch_size=args.test_batch_size,
                                  num_workers=max(1, os.cpu_count() // 2)), 's'))
    if args.mode in ['tail', 'both', 'full']:
        ts_dls.append((DataLoader(TestDataset(ts_q, al_q, ev_ix, 'o', args),
                                  batch_size=args.test_batch_size,
                                  num_workers=max(1, os.cpu_count() // 2)), 'o'))
    if args.mode in ['time', 'full']:
        ts_dls.append((DataLoader(TestDataset(ts_q, al_q, ev_ix, 't', args),
                                  batch_size=args.test_batch_size,
                                  num_workers=max(1, os.cpu_count() // 2)), 't'))

    logs = []
    stp = 1
    tot_stp = args.valid_approximation or sum([len(ts_dl) for ts_dl, _ in ts_dls])
    with torch.no_grad():
        for ts_dl, md in ts_dls:
            for pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel, fil_b in ts_dl:
                if stp > tot_stp:
                    break
                if args.cuda:
                    pos = pos.cuda()
                    neg = neg.cuda()
                    neg_abs = neg_abs.cuda()
                    neg_abs_s_rel = neg_abs_s_rel.cuda()
                    neg_abs_o_rel = neg_abs_o_rel.cuda()
                    neg_rel = neg_rel.cuda()
                    fil_b = fil_b.cuda()

                sc = mdl((pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel), md)
                sc += fil_b * torch.max(sc.abs(), dim=1).values.view(-1, 1)
                as_sc = torch.argsort(sc, dim=1, descending=True).cpu().numpy()

                if md == 's':
                    true_pos, pos_u_ix = pos[:, 0], pos[:, 2]
                elif md == 'o':
                    true_pos, pos_u_ix = pos[:, 2], pos[:, 0]
                elif md == 't':
                    true_pos = []
                    for (d, m, y) in pos[:, 3:6]:
                        for i, t in enumerate(ts_dl.dataset.ts):
                            if (t.day == d and t.month == m and t.year == y):
                                true_pos.append(i)
                    true_pos = torch.from_numpy(np.array(true_pos).astype(np.int))

                for i in range(pos.size(0)):
                    r = np.argwhere(as_sc[i, :] == true_pos[i].item())[0, 0] + 1
                    if md != 't' and args.type_evaluation:
                        ix = tp_ix[tp_rix[true_pos[i].item()]]
                        if args.heuristic_evaluation:
                            _ix = u_ix.get(e_ix.get(pos_u_ix[i].item(), ''), [])
                            u_r = np.isin(as_sc[i, :], _ix)[:r].sum()
                            r = np.isin(as_sc[i, :], ix)[:r].sum() if u_r == 0 else u_r
                        else:
                            r = np.isin(as_sc[i, :], ix)[:r].sum()

                    logs.append({'MRR': 1.0 / r,
                                 'MR': float(r),
                                 'H1': 1.0 if r <= 1 else 0.0,
                                 'H3': 1.0 if r <= 3 else 0.0,
                                 'H10': 1.0 if r <= 10 else 0.0, })

                if stp % args.test_log_steps == 0:
                    logging.info(f'Evaluating the model ... ({stp}/{tot_stp})')

                stp += 1

    mtrs = {}
    for mtr in logs[0].keys():
        mtrs[mtr] = sum([log[mtr] for log in logs]) / len(logs)

    return mtrs
