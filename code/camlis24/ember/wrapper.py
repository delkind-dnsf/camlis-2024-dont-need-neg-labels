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
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..ember.loader import PuEmberDfDataset, EmberCollate
from ..label_corruption import pu_label_corruption
from ..nn_utils import count_params, worker_init_fn
from ..pipeline import AbstractHarness, get_test_metrics


def ember_cross_val_wrapper(
    train_val_df, test_df, net_class, pipeline_class: AbstractHarness, args
):
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    results = []
    for fold, (train_ndx, val_ndx) in enumerate(
        kf.split(range(len(train_val_df)), train_val_df["label"])
    ):
        torch.manual_seed(args.seed + fold)
        df_train = train_val_df.iloc[train_ndx].copy(deep=True)
        df_val = train_val_df.iloc[val_ndx].copy(deep=True)
        assert len(df_train) >= len(
            df_val
        ), f"len(df_train)={len(df_train)}, len(df_val)={len(df_val)}"

        # Do the re-labeling
        df_train = pu_label_corruption(
            df_train, corrupt_frac=args.corrupt_frac, random_seed=args.seed
        )
        df_val = pu_label_corruption(
            df_val, corrupt_frac=args.corrupt_frac, random_seed=args.seed
        )
        test_df["pu_label"] = test_df["label"]

        # Print informational messages
        n_unlab = (
            0
            if args.exclude_unlabeled
            else len(df_train.loc[(df_train["label"] == -1)])
        )
        n_train = len(df_train.loc[df_train["label"].isin([0, 1])]) + n_unlab

        n_fn = 0.5 * n_unlab + len(
            df_train.loc[(df_train["label"] == 1) & (df_train["pu_label"] == 0)]
        )
        n_neg = (
            len(
                df_train.loc[
                    (df_train["label"].isin([0, 1])) & (df_train["pu_label"] == 0)
                ]
            )
            + n_unlab
        )
        args.alpha_true = n_fn / n_neg
        print(
            " ".join(
                [
                    f"There are {n_train:,} training samples",
                    f"of which {n_neg:,} are unlabeled"
                    f"and {n_fn:,} of the unlabeled set are positives,",
                    f"so the true mixture percentage is {100. *args.alpha_true:.2f}.",
                ]
            )
        )

        assert "idx" in df_train.columns
        dataset_train = PuEmberDfDataset(
            df_train, subset="train", exclude_unlabeled=args.exclude_unlabeled
        )
        dataset_valid = PuEmberDfDataset(
            df_val, subset="train", exclude_unlabeled=args.exclude_unlabeled
        )
        dataset_test = PuEmberDfDataset(
            test_df, subset="test", exclude_unlabeled=args.exclude_unlabeled
        )
        print(
            f"Train size: {len(dataset_train):,}\nValidation size: {len(dataset_valid):,}"
        )

        collate_fn = EmberCollate()

        train_q = DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        val_q = DataLoader(
            dataset=dataset_valid,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
            drop_last=False,
            shuffle=True,
        )
        test_q = DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=True,
            drop_last=False,
            shuffle=True,
        )

        center, scale = dataset_train.get_center_scale()
        args.net_config.update({"center": center, "scale": scale})

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
            train_queue=train_q,
            val_queue=val_q,
            n_warmup_epoch=args.n_warmup_epoch,
            n_epoch=args.n_epoch,
            net=net,
            optim=optim,
        )
        args.tb_writer.close()

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
