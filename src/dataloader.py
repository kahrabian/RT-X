import bisect
import calendar
import pytz
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    @staticmethod
    def _frq(tup, start=4):
        frq = {}
        for s, r, o, _ in tup:
            if (s, r) not in frq:
                frq[(s, r)] = start
            else:
                frq[(s, r)] += 1

            if (o, -r - 1) not in frq:
                frq[(o, -r - 1)] = start
            else:
                frq[(o, -r - 1)] += 1
        return frq

    @staticmethod
    def _true_s_o(tup):
        true_s = {}
        true_o = {}

        for s, r, o, _ in tup:
            if (s, r) not in true_o:
                true_o[(s, r)] = []
            true_o[(s, r)].append(o)
            if (r, o) not in true_s:
                true_s[(r, o)] = []
            true_s[(r, o)].append(s)

        for r, o in true_s:
            true_s[(r, o)] = np.array(list(set(true_s[(r, o)])))
        for s, r in true_o:
            true_o[(s, r)] = np.array(list(set(true_o[(s, r)])))

        return true_s, true_o

    def __init__(self, tup, tp_ix, tp_rix, ev_ix, md, args):
        self.tup = tup
        self.md = md

        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.neg_sz = args.negative_sample_size
        self.neg_t_sz = args.negative_time_sample_size

        self.frq = self._frq(tup)
        self.true_s, self.true_o = self._true_s_o(self.tup)

        self.neg_tp_smpl = args.negative_type_sampling

        self.tp_ix = tp_ix
        self.tp_rix = tp_rix
        self.ev_ix = ev_ix

        self.tz = pytz.timezone(args.timezone)
        self.min_ts = min(map(lambda x: x[3], self.tup))
        self.max_ts = max(map(lambda x: x[3], self.tup))

        self.max_t_gap = args.negative_max_time_gap

    def __len__(self):
        return len(self.tup)

    def _lt(self, e, ts, t_gap=None):
        t_gap = np.random.randint(0, self.max_t_gap + 1) if t_gap is None else t_gap
        ts -= t_gap
        rel_ts = []
        for i in range(self.nrelation):
            rel_ev_ix = bisect.bisect_left(self.ev_ix[e][i], ts) - 1
            rel_ts.append((ts - (self.ev_ix[e][i][rel_ev_ix] if rel_ev_ix != -1 else self.min_ts)) // (24 * 60 * 60))
        return rel_ts

    def _neg(self, s, r, o, t):
        neg_list = []
        neg_size = 0

        while neg_size < self.neg_sz:
            if not self.neg_tp_smpl:
                neg = np.random.randint(self.nentity, size=self.neg_sz * 2)
            else:
                if self.md == 's':
                    ss = self.tp_ix[self.tp_rix[s]]
                elif self.md == 'o':
                    ss = self.tp_ix[self.tp_rix[o]]
                neg = np.random.choice(ss, size=self.neg_sz * 2)
            if self.md == 's':
                msk = np.in1d(neg, self.true_s[(r, o)], assume_unique=True, invert=True)
            elif self.md == 'o':
                msk = np.in1d(neg, self.true_o[(s, r)], assume_unique=True, invert=True)
            neg = neg[msk]
            neg_list.append(neg)
            neg_size += neg.size

        neg = np.array([], dtype=np.int64)
        neg_rel = np.array([], dtype=np.int64)
        if len(neg_list) != 0:
            neg = np.concatenate(neg_list)[:self.neg_sz]
            neg_rel = np.apply_along_axis(lambda x: self._lt(x[0], t), 0, neg.reshape(1, -1)).T

        return torch.from_numpy(neg), torch.from_numpy(neg_rel)

    def _neg_abs(self, s, o, d, m, y):
        neg_t_ls = []
        neg_t_sz = 0

        while neg_t_sz < self.neg_t_sz:
            neg_t = np.random.randint(self.min_ts, self.max_ts + 1, size=self.neg_t_sz * 2)
            neg_d = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).day, 0, neg_t.reshape(1, -1))
            neg_m = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).month, 0, neg_t.reshape(1, -1))
            neg_y = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).year, 0, neg_t.reshape(1, -1))

            neg_t = neg_t[(neg_d != d) | (neg_m != m) | (neg_y != y)]
            neg_t_ls.append(neg_t)
            neg_t_sz += neg_t.size

        neg_abs = np.array([[], [], []], dtype=np.int64)
        neg_abs_s_rel = np.array([], dtype=np.int64)
        neg_abs_o_rel = np.array([], dtype=np.int64)
        if len(neg_t_ls) != 0:
            neg_t = np.concatenate(neg_t_ls)[:self.neg_t_sz]
            neg_abs_d = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).day, 0, neg_t.reshape(1, -1))
            neg_abs_m = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).month, 0, neg_t.reshape(1, -1))
            neg_abs_y = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).year, 0, neg_t.reshape(1, -1))
            neg_abs = np.stack([neg_abs_d, neg_abs_m, neg_abs_y])
            neg_abs_s_rel = np.apply_along_axis(lambda x: self._lt(s, x[0]), 0, neg_t.reshape(1, -1)).T
            neg_abs_o_rel = np.apply_along_axis(lambda x: self._lt(o, x[0]), 0, neg_t.reshape(1, -1)).T

        return torch.from_numpy(neg_abs), torch.from_numpy(neg_abs_s_rel), torch.from_numpy(neg_abs_o_rel)

    def __getitem__(self, ix):
        s, r, o, t = self.tup[ix]
        d = datetime.fromtimestamp(t, self.tz).day
        m = datetime.fromtimestamp(t, self.tz).month
        y = datetime.fromtimestamp(t, self.tz).year

        t_gap = np.random.randint(0, self.max_t_gap + 1)
        s_rel = self._lt(s, t, t_gap=t_gap)
        o_rel = self._lt(o, t, t_gap=t_gap)

        smpl_w = torch.sqrt(1 / torch.Tensor([self.frq[(s, r)] + self.frq[(o, (-1) * r - 1)], ]))

        neg, neg_rel = self._neg(s, r, o, t)
        neg_abs, neg_abs_s_rel, neg_abs_o_rel = self._neg_abs(s, o, d, m, y)

        pos = torch.LongTensor([s, r, o, d, m, y, *s_rel, *o_rel])

        return pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel, smpl_w


class TestDataset(Dataset):
    def __init__(self, tup, al_t, ev_ix, md, args):
        self.tup = tup
        self.md = md

        self.nentity = args.nentity
        self.nrelation = args.nrelation

        self.ev_ix = ev_ix

        self.tz = pytz.timezone(args.timezone)
        self.min_ts = min(map(lambda x: x[3], self.tup))
        self.max_ts = max(map(lambda x: x[3], self.tup))
        self.ts = pd.date_range(datetime.fromtimestamp(self.min_ts, self.tz).strftime('%Y-%m-%d'),
                                datetime.fromtimestamp(self.max_ts, self.tz).strftime('%Y-%m-%d'), tz=self.tz)

        self.al_t = set()
        for t in al_t:
            ts = datetime.fromtimestamp(t[3], self.tz)
            self.al_t.add((t[0], t[1], t[2], ts.day, ts.month, ts.year))

        self.max_t_gap = args.negative_max_time_gap

    def __len__(self):
        return len(self.tup)

    def _lt(self, e, ts, t_gap=None):
        t_gap = np.random.randint(0, self.max_t_gap + 1) if t_gap is None else t_gap
        ts -= t_gap
        rel_ts = []
        for i in range(self.nrelation):
            rel_ev_ix = bisect.bisect_left(self.ev_ix[e][i], ts) - 1
            rel_ts.append((ts - (self.ev_ix[e][i][rel_ev_ix] if rel_ev_ix != -1 else self.min_ts)) // (24 * 60 * 60))
        return rel_ts

    def __getitem__(self, ix):
        s, r, o, t = self.tup[ix]
        d = datetime.fromtimestamp(t, self.tz).day
        m = datetime.fromtimestamp(t, self.tz).month
        y = datetime.fromtimestamp(t, self.tz).year

        t_gap = np.random.randint(0, self.max_t_gap + 1)
        s_rel = self._lt(s, t, t_gap=t_gap)
        o_rel = self._lt(o, t, t_gap=t_gap)

        if self.md == 's':
            fil_b_neg = [(0, i) if (i, r, o, d, m, y) not in self.al_t else (-1, s) for i in range(self.nentity)]
            fil_b_neg[s] = (0, s)
        elif self.md == 'o':
            fil_b_neg = [(0, i) if (s, r, i, d, m, y) not in self.al_t else (-1, o) for i in range(self.nentity)]
            fil_b_neg[o] = (0, o)
        elif self.md == 't':
            fil_b_neg = []
            for t in self.ts:
                if (s, r, o, t.day, t.month, t.year) not in self.al_t or (t.day == d and t.month == m and t.year == y):
                    fil_b_neg.append((0, t.timestamp()))
                else:
                    fil_b_neg.append((-1, t.timestamp()))

        fil_b_neg = torch.LongTensor(fil_b_neg)
        fil_b = fil_b_neg[:, 0].float()
        if self.md != 't':
            neg = fil_b_neg[:, 1]
            neg_abs = torch.from_numpy(np.array([[], [], []], dtype=np.int64))

            neg_rel = torch.from_numpy(np.apply_along_axis(lambda x: self._lt(x[0], t), 0, neg.reshape(1, -1)).T)
            neg_abs_s_rel = torch.from_numpy(np.array([], dtype=np.int64))
            neg_abs_o_rel = torch.from_numpy(np.array([], dtype=np.int64))
        else:
            neg = torch.from_numpy(np.array([], dtype=np.int64))
            neg_t = fil_b_neg[:, 1]
            neg_abs_d = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).day, 0, neg_t.reshape(1, -1))
            neg_abs_m = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).month, 0, neg_t.reshape(1, -1))
            neg_abs_y = np.apply_along_axis(lambda x: datetime.fromtimestamp(x[-1], self.tz).year, 0, neg_t.reshape(1, -1))
            neg_abs = np.stack([neg_abs_d, neg_abs_m, neg_abs_y])

            neg_rel = torch.from_numpy(np.array([], dtype=np.int64))
            neg_abs_s_rel = torch.from_numpy(np.apply_along_axis(lambda x: self._lt(s, x[0]), 0, neg_t.reshape(1, -1)).T)
            neg_abs_o_rel = torch.from_numpy(np.apply_along_axis(lambda x: self._lt(o, x[0]), 0, neg_t.reshape(1, -1)).T)

        pos = torch.LongTensor((s, r, o, d, m, y, *s_rel, *o_rel))

        return pos, neg, neg_abs, neg_abs_s_rel, neg_abs_o_rel, neg_rel, fil_b


class BidirectionalOneShotIterator(object):
    @staticmethod
    def it(dl):
        while True:
            for d in dl:
                yield d

    def __init__(self, dl_s, dl_o):
        self.stp = 1
        self.it_s = self.it(dl_s)
        self.it_o = self.it(dl_o)

    def __next__(self):
        self.stp += 1
        if self.stp % 2 == 0:
            return next(self.it_s) + ['s', ]
        else:
            return next(self.it_o) + ['o', ]
