import os
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image

from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import histoset_dataroot


class HistoSetDataset14(Dataset):
    """
    CSV-driven dataset for HistoSet-5x14.
    Must return: img, label, uq_idx  (so that MergedDataset can add mask_lab)
    CSV columns supported:
      - path,label
      - img_path,label
    """

    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform

        df = pd.read_csv(csv_path)
        if "path" in df.columns:
            path_col = "path"
        elif "img_path" in df.columns:
            path_col = "img_path"
        else:
            raise ValueError(f"CSV must contain 'path' or 'img_path'. Got: {df.columns.tolist()}")
        if "label" not in df.columns:
            raise ValueError(f"CSV must contain 'label'. Got: {df.columns.tolist()}")

        self.samples = []
        for p, y in zip(df[path_col].tolist(), df["label"].tolist()):
            abs_path = os.path.join(self.root, str(p))
            self.samples.append([abs_path, int(y)])

        self.targets = [int(x[1]) for x in self.samples]
        self.uq_idxs = np.array(range(len(self)))

        # allow outside to assign
        self.target_transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        uq_idx = int(self.uq_idxs[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(int(label))

        return img, int(label), uq_idx


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype("bool")
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples, dtype=object)[mask].tolist()
    dataset.targets = np.array(dataset.targets, dtype=object)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]
    return dataset


def subsample_classes(dataset, include_classes):
    """
    Keep only instances whose ORIGINAL label is in include_classes.
    We do NOT rewrite dataset.targets; we only subset samples (labels stay original ids).
    """
    include_classes = list(include_classes)
    cls_idxs = [i for i, l in enumerate(dataset.targets) if int(l) in set(include_classes)]
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset


def get_histoset_datasets(
        train_transform,
        test_transform,
        train_classes,
        prop_train_labels=0.5,
        seed=0,
        split_train_val=False,
        val_instances_per_class=5,
        train_csv_name="histoset5x14_label_20p.csv",
        test_csv_name=None,
):
    """
    Return dict with keys: train_labelled, train_unlabelled, val, test
    Follow the same high-level logic used in Selex gastrovision dataset loader. :contentReference[oaicite:4]{index=4}
    """
    np.random.seed(seed)

    train_csv_path = os.path.join(histoset_dataroot, train_csv_name)
    if test_csv_name is None:
        test_csv_path = train_csv_path
    else:
        test_csv_path = os.path.join(histoset_dataroot, test_csv_name)

    # Full pool
    train_dataset = HistoSetDataset14(
        root=histoset_dataroot,
        csv_path=train_csv_path,
        transform=train_transform
    )

    # Labelled = known classes, then pick prop_train_labels portion as "labelled"
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(
        train_dataset_labelled,
        prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Unlabelled = everything else in the pool
    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    # Test set (same CSV by default)
    test_dataset = HistoSetDataset14(
        root=histoset_dataroot,
        csv_path=test_csv_path,
        transform=test_transform
    )

    all_datasets = {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": None,
        "test": test_dataset,
    }
    return all_datasets