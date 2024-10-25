#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-29 (yyyy-mm-dd)

"""
"""
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..dga.loader import PuDgaDataset, IntCollatePad
from ..label_corruption import pu_label_corruption
from ..nn_utils import count_params, worker_init_fn
from ..pipeline import AbstractHarness, get_test_metrics


def dga_cross_val_wrapper(df, net_class, pipeline_class: AbstractHarness, args):
    if args.dry_run:
        df = df.sample(int(0.25 * len(df)), random_state=args.seed)

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    df["label"] = df["class"].apply(lambda x: 0 if x == 0 else 1)  # binary classes
    results = []
    for fold, (train_val_ndx, test_ndx) in enumerate(kf.split(df, df["label"])):
        torch.manual_seed(args.seed + fold)
        print(20 * "=" + f"\tFold {fold}\t" + 20 * "=")
        df_test = df.iloc[test_ndx].copy(deep=True)
        df_test["pu_label"] = df_test["label"]

        train_ndx, val_ndx = train_test_split(
            train_val_ndx,
            test_size=1.0 / args.n_splits,
            stratify=df.iloc[train_val_ndx]["label"],
            random_state=args.seed,
        )
        df_train = df.iloc[train_ndx].copy(deep=True)
        df_train = pu_label_corruption(
            df_train, corrupt_frac=args.corrupt_frac, random_seed=args.seed
        )
        n_fn = len(df_train.loc[(df_train["label"] == 1) & (df_train["pu_label"] == 0)])
        n_neg = len(df_train.loc[(df_train["pu_label"] == 0)])
        args.alpha_true = n_fn / n_neg
        print(
            " ".join(
                [
                    f"There are {len(df_train):,} training samples",
                    f"of which {n_neg:,} are unlabeled"
                    f"and {n_fn:,} of the unlabeled set are positive,",
                    f"so the true mixture proportion is α={100. * args.alpha_true:.2f} percent.",
                ]
            )
        )
        dataset_train = PuDgaDataset(df_train, codec=args.codec)

        df_val = df.iloc[val_ndx].copy(deep=True)
        df_val = pu_label_corruption(
            df_val, corrupt_frac=args.corrupt_frac, random_seed=args.seed
        )
        dataset_valid = PuDgaDataset(df_val, codec=args.codec)
        collate_fn = IntCollatePad(max_len=args.max_len, pad_index=args.pad_index)
        train_q = DataLoader(
            dataset=dataset_train,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=True,
            shuffle=True,
        )
        val_q = DataLoader(
            dataset=dataset_valid,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=False,
            shuffle=True,
        )

        net = net_class(**args.net_config)
        print(f"net has {count_params(net):,} parameters")
        net.to(device=args.device)

        model_fold_name = args.model_name + f"-fold_{fold}"
        args.tb_writer = SummaryWriter(
            log_dir=args.save_point.joinpath(model_fold_name)
        )

        optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pipeline = pipeline_class(args=args)
        net, _ = pipeline.main(
            n_warmup_epoch=args.n_warmup_epoch,
            n_epoch=args.n_epoch,
            train_queue=train_q,
            val_queue=val_q,
            net=net,
            optim=optim,
        )

        dataset_test = PuDgaDataset(df_test, codec=args.codec)
        test_q = DataLoader(
            dataset=dataset_test,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=False,
            shuffle=True,
        )
        scores = get_test_metrics(test_q, net=net, device=args.device)
        accuracy, brier_score, cross_entropy, roc_auc, trim_cross_entropy = scores
        new_results = {
            "fold": fold,
            "accuracy": accuracy,
            "brier_score": brier_score,
            "cross_entropy": cross_entropy,
            "trim_cross_entropy": trim_cross_entropy,
            "roc_auc": roc_auc,
            "alpha_true": args.alpha_true,
            "corrupt_frac": args.corrupt_frac,
        }
        results.append(new_results)
        for k, v in new_results.items():
            if k == "fold":
                continue
            print(f"{k}:\t{v:.4f}")
        args.tb_writer.close()
        if args.dry_run and fold >= 1:
            break
    results_df = pd.DataFrame(results)
    results_df["model"] = args.model_name
    return results_df
