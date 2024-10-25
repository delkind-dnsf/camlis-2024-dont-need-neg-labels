#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David Elkind
# Organization: DNSFilter
# (c) 2024 DNS Filter. All rights reserved.
# Creation date: 2024-08-19 (yyyy-mm-dd)

"""
"""
import datetime
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import xlogy
from scipy.stats import trim_mean
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dga.loader import PuDgaDataset, IntCollatePad
from .ember.loader import PuEmberDfDataset, EmberCollate
from .nn_utils import worker_init_fn
from .ted_n import bbe_estimate, SmoothedEcdf


def get_model_id(inp):
    fname = pathlib.Path(inp).stem
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    return "_".join([fname, now])


def get_checkpoint_name(args, epoch):
    dest_dir = args.save_point.joinpath("checkpoint")
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{args.model_name}_E{epoch:03}.pth"
    return dest_dir.joinpath(fname)


def save_checkpoint(args, epoch, net, optim, epoch_val_loss, model_config=None):
    destination = get_checkpoint_name(args, epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch_val_loss": epoch_val_loss,
            "model_config": model_config,
        },
        destination,
    )
    return destination


class MeanStatBuffer:
    def __init__(self, tag, buff_size, tensorboard):
        assert isinstance(buff_size, int) and buff_size > 0
        self._offset = 0
        self._buff = np.zeros(buff_size)
        self.tb_writer = tensorboard
        self.tag = tag

    @property
    def offset(self):
        return self._offset

    @property
    def buff(self):
        return self._buff

    @property
    def stat(self):
        return self.buff.mean()

    def append(self, x, global_step=None, **kwargs):
        assert isinstance(x, (float, int))
        self._buff[self.offset % self.buff.size] = x
        self._offset += 1
        if self.offset % self._buff.size == 0 and 0 < self.offset:
            global_step = self.offset if global_step is None else 0
            self.write(stat=self.stat, global_step=global_step, **kwargs)

    def write(self, stat, global_step, **kwargs):
        self.tb_writer.add_scalar(self.tag, stat, global_step=global_step, **kwargs)


class Percentile95Buffer(MeanStatBuffer):
    @property
    def stat(self):
        return np.percentile(self.buff, q=95)


class EarlyStopping:
    def __init__(self, patience, min_improvement=np.sqrt(2.0**-23.0), verbose=False):
        self.best_loss = float("inf")
        self.patience = patience
        self.best_epoch = 0
        self.epoch_counter = -1
        self.verbose = verbose
        self.min_improvement = min_improvement

    def __call__(self, new_loss):
        self.epoch_counter += 1
        if new_loss + self.min_improvement < self.best_loss:
            old_best_loss = self.best_loss
            self.best_loss = new_loss
            self.best_epoch = self.epoch_counter
            if self.verbose:
                print(
                    f"\t\tNew best_val_loss: {self.best_loss:.4f} (Old best loss: {old_best_loss:.4f} at epoch {self.best_epoch}, improvement: {old_best_loss-self.best_loss})"
                )
        if self.epoch_counter >= self.patience + self.best_epoch:
            print(
                f"\t\tEarly stopping patience expired after {self.patience} epochs. Best epoch: {self.best_epoch} with loss {self.best_loss:.4f}"
            )
            return True
        return False


class AbstractHarness:
    def __init__(self, args):
        self.args = args
        self.loss_callable = nn.BCEWithLogitsLoss(reduction="mean")

        self.early_stop = EarlyStopping(
            patience=self.args.early_stop_patience, min_improvement=1e-3
        )
        self.tb_train_recent = MeanStatBuffer(
            tag=f"train/recent batch loss avg",
            buff_size=32,
            tensorboard=self.args.tb_writer,
        )
        self.tb_train_err_epoch = MeanStatBuffer(
            tag=f"train/epoch Brier score avg",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.tb_train_batch = MeanStatBuffer(
            tag=f"train/batch loss",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.tb_train_upper = Percentile95Buffer(
            tag=f"train/95th percentile",
            buff_size=128,
            tensorboard=self.args.tb_writer,
        )
        self.tb_grad_norm = MeanStatBuffer(
            tag="train/grad norm", buff_size=32, tensorboard=self.args.tb_writer
        )
        self.tb_val_epoch = MeanStatBuffer(
            tag=f"val/epoch Brier score avg",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )

    def loss_fn(self, logits, labels, *args, **kwargs):
        return self.loss_callable(logits, labels)

    def log_params(self, net, epoch):
        for name, param in net.named_parameters():
            self.args.tb_writer.add_histogram(name, param, epoch)

    @staticmethod
    def val_loss_fn(proba_predictions, labels, *args, **kwargs):
        return brier_score_loss(labels, proba_predictions)

    def train_epoch(self, train_queue, net, optim):
        net.train()
        for i, batch_data in enumerate(tqdm(train_queue)):
            optim.zero_grad()
            x, y, *_ = batch_data
            x, y = [item.to(self.args.device) for item in (x, y)]
            logits = net(x)
            loss_val = self.loss_fn(logits, y)
            loss_val.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), self.args.max_grad_norm
            )
            optim.step()
            self.tb_grad_norm.append(grad_norm.item())
            self.tb_train_recent.append(loss_val.item())
            self.tb_train_batch.append(loss_val.item())
            self.tb_train_upper.append(loss_val.item())

    @torch.no_grad()
    def validation_batch(self, batch_data, net):
        net.eval()
        x, y, *_ = batch_data
        x, y = [item.to(self.args.device) for item in (x, y)]
        return net.predict_proba(x)


class PosNegHarnessFcn(AbstractHarness):
    def __init__(self, args):
        super(PosNegHarnessFcn, self).__init__(args=args)

    @torch.no_grad()
    def train_queue_callback(self, train_queue, net, class_prior=0.0):
        net.eval()
        df_list = []
        for i, batch_data in enumerate(tqdm(train_queue)):
            _, y, batch_df = batch_data
            outputs = self.validation_batch(batch_data, net)
            batch_df["pr_malicious"] = outputs.numpy(force=True)
            df_list.append(batch_df)

        all_preds_df = pd.concat(df_list)
        epoch_loss = self.val_loss_fn(
            all_preds_df["pr_malicious"], all_preds_df["pu_label"]
        )
        return epoch_loss, train_queue

    @torch.no_grad()
    def validation_epoch(self, val_queue, net, *args, **kwargs):
        net.eval()
        val_preds = []
        val_label = []
        for i, batch_data in enumerate(tqdm(val_queue)):
            _, labels, *_ = batch_data
            outputs = self.validation_batch(batch_data, net)
            val_preds.append(outputs)
            val_label.append(labels)
        val_preds = torch.cat(val_preds).numpy(force=True)
        val_label = torch.cat(val_label).numpy(force=True)
        return self.val_loss_fn(proba_predictions=val_preds, labels=val_label)

    def main(self, n_warmup_epoch, n_epoch, train_queue, val_queue, net, optim):
        scheduler = StepLR(
            optim,
            step_size=self.args.lr_scheduler_step_size,
            gamma=self.args.lr_scheduler_gamma,
        )
        for epoch in range(n_epoch + n_warmup_epoch):
            self.train_epoch(train_queue, net, optim)
            scheduler.step()
            val_loss = self.validation_epoch(val_queue, net)
            self.tb_val_epoch.append(val_loss)
            train_loss, _ = self.train_queue_callback(train_queue, net)
            self.tb_train_err_epoch.append(train_loss)
            save_checkpoint(
                args=self.args,
                epoch=epoch,
                net=net,
                optim=optim,
                epoch_val_loss=val_loss,
            )
            self.log_params(net, epoch)
            if self.early_stop(val_loss):
                # The TEDn procedure stops training when 'the training error converges',
                # whereas the conventional training stops when out-of-sample loss stops
                # improving
                break
        checkpoint = torch.load(
            get_checkpoint_name(self.args, self.early_stop.best_epoch)
        )
        net.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return net, self.early_stop.best_loss


class PosUnlabDgaHarnessFcn(AbstractHarness):
    def __init__(self, args):
        super(PosUnlabDgaHarnessFcn, self).__init__(args=args)
        self.tb_cpe = MeanStatBuffer(
            tag=f"val/â", buff_size=1, tensorboard=self.args.tb_writer
        )
        self.tb_cpe_err = MeanStatBuffer(
            tag=f"val/â - a", buff_size=1, tensorboard=self.args.tb_writer
        )
        self.tb_malicious_excluded = MeanStatBuffer(
            tag=f"TEDⁿ/count FN excluded (thousands)",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.tb_pct_malicious_excluded = MeanStatBuffer(
            tag=f"TEDⁿ/FN excluded (%)",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.tb_benign_excluded = MeanStatBuffer(
            tag=f"TEDⁿ/count unlabeled or negative excluded (thousands)",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.tb_pct_malicious_included = MeanStatBuffer(
            tag=f"TEDⁿ/FN among unlabeled (%)",
            buff_size=1,
            tensorboard=self.args.tb_writer,
        )
        self.class_prior_clip = 0.01

    def main(self, n_warmup_epoch, n_epoch, val_queue, train_queue, net, optim):
        assert n_warmup_epoch > 0
        epoch = 0
        scheduler = StepLR(
            optim,
            step_size=self.args.lr_scheduler_step_size,
            gamma=self.args.lr_scheduler_gamma,
        )
        for i in range(n_warmup_epoch):
            self.train_epoch(train_queue, net, optim)
            scheduler.step()
            val_loss, class_prior = self.validation_epoch(val_queue, net)
            self.tb_cpe.append(class_prior)
            self.tb_cpe_err.append(class_prior - self.args.alpha_true)
            self.tb_val_epoch.append(val_loss)
            save_checkpoint(
                args=self.args,
                epoch=epoch,
                net=net,
                optim=optim,
                epoch_val_loss=val_loss,
            )
            train_loss, pu_queue = self.train_queue_callback(
                train_queue, net, class_prior
            )
            self.tb_train_err_epoch.append(train_loss)
            self.early_stop(train_loss)
            self.log_params(net, epoch)
            epoch += 1

        for i in range(n_warmup_epoch, n_warmup_epoch + n_epoch):
            self.train_epoch(pu_queue, net, optim)
            scheduler.step()
            val_loss, class_prior = self.validation_epoch(val_queue, net)
            self.tb_cpe.append(class_prior)
            self.tb_cpe_err.append(class_prior - self.args.alpha_true)
            self.tb_val_epoch.append(val_loss)
            save_checkpoint(
                args=self.args,
                epoch=epoch,
                net=net,
                optim=optim,
                epoch_val_loss=val_loss,
            )
            train_loss, pu_queue = self.train_queue_callback(
                train_queue, net, class_prior
            )
            self.tb_train_err_epoch.append(train_loss)
            self.log_params(net, epoch)
            epoch += 1
            if self.early_stop(train_loss):
                # The TEDn procedure stops training when 'the training error converges,'
                # whereas the conventional training stops when out-of-sample loss stops
                # improving.
                break

        checkpoint = torch.load(
            get_checkpoint_name(self.args, self.early_stop.best_epoch)
        )
        net.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return net, self.early_stop.best_loss

    @torch.no_grad()
    def validation_epoch(self, val_queue, net, *args, **kwargs):
        # get predictions for validation data
        net.eval()
        val_preds = []
        val_label = []
        for i, batch_data in enumerate(tqdm(val_queue)):
            _, labels, *_ = batch_data
            outputs = self.validation_batch(batch_data, net)
            val_preds.append(outputs)
            val_label.append(labels)
        val_preds = torch.cat(val_preds).numpy(force=True)
        val_label = torch.cat(val_label).numpy(force=True)

        # compute class prior from validation data
        # we can't append within the batch loop, because we can't compute AUC batch-wise
        p_hist = SmoothedEcdf(smoother=self.args.bbe_smoother)
        u_hist = SmoothedEcdf(smoother=0.0)

        p_hist.append(val_preds[np.where(val_label == 1)])
        u_hist.append(val_preds[np.where(val_label == 0)])

        class_prior, _ = bbe_estimate(
            p_hist=p_hist,
            u_hist=u_hist,
            delta=self.args.bbe_delta,
            gamma=self.args.bbe_gamma,
        )
        # values of 0.0 or 1.0 break stuff downstream, so we clip values here
        # we only experiment with noise levels between 0.1 and 0.9, so 0.01 to 0.99
        # avoids the downstream problem while still allowing the estimator to produce a
        # value that is too small/too large
        class_prior = np.median(
            [class_prior, self.class_prior_clip, 1.0 - self.class_prior_clip]
        ).item()

        # get the validation loss metric
        val_epoch_loss = self.val_loss_fn(val_preds, val_label)

        return val_epoch_loss, class_prior

    def make_pu_queue(self, pu_df):
        pu_dataset = PuDgaDataset(df=pu_df, codec=self.args.codec)
        collate_fn = IntCollatePad(
            max_len=self.args.max_len, pad_index=self.args.pad_index
        )
        pu_queue = DataLoader(
            dataset=pu_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.args.seed),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        return pu_queue

    @torch.no_grad()
    def train_queue_callback(self, train_queue, net, class_prior):
        net.eval()
        df_list = []
        for i, batch_data in enumerate(tqdm(train_queue)):
            _, y, batch_df = batch_data
            outputs = self.validation_batch(batch_data, net)
            batch_df["pr_malicious"] = outputs.numpy(force=True)
            df_list.append(batch_df)

        all_preds_df = pd.concat(df_list)
        # get the cutoff for exclusion from training data
        cutoff = all_preds_df.loc[
            all_preds_df["pu_label"] == 0, "pr_malicious"
        ].quantile(1.0 - class_prior)

        # This is some interesting logging stuff for the PU model:
        # among the removed unlabeled data, what proportion are truly malicious?
        fn_removed = len(
            all_preds_df.loc[
                (all_preds_df["label"] == 1)
                & (all_preds_df["pu_label"] == 0)
                & (all_preds_df["pr_malicious"] >= cutoff),
            ]
        )
        all_fn = len(
            all_preds_df.loc[
                (all_preds_df["label"] == 1) & (all_preds_df["pu_label"] == 0)
            ]
        )
        self.tb_malicious_excluded.append(fn_removed / 1e3)
        if all_fn > 0:
            self.tb_pct_malicious_excluded.append(100.0 * fn_removed / all_fn)

        # Among the removed unlabeled data, what proportion are truly benign?
        unlabeled_removed = len(
            all_preds_df.loc[
                (all_preds_df["label"] < 1)
                & (all_preds_df["pu_label"] == 0)
                & (all_preds_df["pr_malicious"] >= cutoff),
            ]
        )
        self.tb_benign_excluded.append(unlabeled_removed / 1e3)
        # in the new dataset, what percentage of samples are malicious?
        malicious_incl = len(
            all_preds_df.loc[
                (all_preds_df["label"] == 1)
                & (all_preds_df["pu_label"] == 0)
                & (all_preds_df["pr_malicious"] < cutoff)
            ]
        )
        unlabeled_incl = len(
            all_preds_df.loc[
                (all_preds_df["pu_label"] == 0)
                & (all_preds_df["pr_malicious"] < cutoff)
            ]
        )
        self.tb_pct_malicious_included.append(100.0 * malicious_incl / unlabeled_incl)

        # get our new dataset
        # remove the unlabeled with the highest pr_malicious
        pu_df = all_preds_df.loc[
            (all_preds_df["pr_malicious"] < cutoff) | (all_preds_df["pu_label"] == 1)
        ]
        pu_queue = self.make_pu_queue(pu_df=pu_df)
        # get the error rate for the epoch
        epoch_loss = self.val_loss_fn(pu_df["pr_malicious"], pu_df["pu_label"])
        return epoch_loss, pu_queue


class PosUnlabEmberHarnessFcn(PosUnlabDgaHarnessFcn):
    def make_pu_queue(self, pu_df):
        pu_dataset = PuEmberDfDataset(
            df=pu_df, subset="train", exclude_unlabeled=self.args.exclude_unlabeled
        )
        collate_fn = EmberCollate()
        pu_queue = DataLoader(
            dataset=pu_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.args.seed),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        return pu_queue


@torch.no_grad()
def get_test_metrics(test_queue, net, device):
    # get predictions for test data
    net.eval()
    val_preds = []
    val_label = []
    for i, batch_data in enumerate(tqdm(test_queue)):
        x, y, *_ = batch_data
        x, y = [t.to(device) for t in (x, y)]
        outputs = net.predict_proba(x)
        val_preds.append(outputs)
        val_label.append(y)
    val_preds = torch.cat(val_preds).numpy(force=True)
    val_label = torch.cat(val_label).numpy(force=True)

    brier_score = brier_score_loss(val_label, val_preds)
    roc_auc = roc_auc_score(val_label, val_preds)
    cross_entropy = log_loss(val_label, val_preds)
    accuracy = accuracy_score(val_label, np.where(val_preds > 0.5, 1, 0))

    val_preds_2col = np.stack([val_preds, 1.0 - val_preds], axis=1)
    eps = np.finfo(val_preds.dtype).eps
    val_preds_2col = np.clip(val_preds_2col, a_min=eps, a_max=1.0 - eps)
    val_label_2col = np.stack([val_label, 1.0 - val_label], axis=1)
    my_cross_entropy_arr = -1.0 * xlogy(val_label_2col, val_preds_2col).sum(axis=1)
    # below: trim 1% of the values, then compute cross-entropy. The hypothesis here is that
    # there are a small number of test set items that the PU model fits poorly, and
    # trimming them will lower the PU model's score compared to the trimmed PUN score.
    trimmed_cross_entropy = trim_mean(my_cross_entropy_arr, proportiontocut=0.05 / 2.0)

    return accuracy, brier_score, cross_entropy, roc_auc, trimmed_cross_entropy
