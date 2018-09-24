#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import timeit
from torch.utils.data import DataLoader
import gc

_lr_multiplier = 0.01


# def ranking(types, model, distfn):
#     lt = th.from_numpy(model.embedding())
#     embedding = Variable(lt, volatile=True)
#     ranks = []
#     ap_scores = []
#     for s, s_types in types.items():
#         s_e = Variable(lt[s].expand_as(embedding), volatile=True)
#         _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
#         _dists[s] = 1e+12
#         _labels = np.zeros(embedding.size(0))
#         _dists_masked = _dists.copy()
#         _ranks = []
#         for o in s_types:
#             _dists_masked[o] = np.Inf
#             _labels[o] = 1
#         ap_scores.append(average_precision_score(_labels, -_dists))
#         for o in s_types:
#             o = o.item()
#             d = _dists_masked.copy()
#             d[o] = _dists[o]
#             r = np.argsort(d)
#             _ranks.append(np.where(r == o)[0][0] + 1)
#         ranks += _ranks
#     # print (np.mean(ranks), np.mean(ap_scores))
#     return np.mean(ranks), np.mean(ap_scores)


def train_mp(model, data, optimizer, opt, log, rank, queue):
    try:
        train(model, data, optimizer, opt, log, rank, queue)
    except Exception as err:
        log.exception(err)
        queue.put(None)


def train(model, data, optimizer, opt, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )
    for epoch in range(opt.epochs):
        epoch_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burnin: lr={lr}')
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data.item())

        if rank == 1:
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model
            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb)
                )
            else:
                log.info(
                    'info: {'
                    f'"elapsed": {elapsed}, '
                    f'"loss": {np.mean(epoch_loss)}, '
                    '}'
                )
        gc.collect()
        print (f"done epoch {epoch} loss: {loss.data.item()}")
        # mean_rank, mAP = ranking(types, model, distfn)
        np.savetxt(X=model.lt.weight.detach().numpy(), fname="cora_ml.embedding", delimiter=",")
