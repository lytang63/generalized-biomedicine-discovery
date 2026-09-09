import os
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import gastrovision_dataroot


class GastrovisionDataset27(Dataset):
    """
    CSV-driven dataset.
    Must return: img, label, uq_idx  (keep same as HerbariumDataset19)
    """

    def __init__(self, root, csv_path, transform=None):
        """
        Args:
            root: dataset root directory
            csv_path: CSV file path with columns [path, label]
            transform: torchvision-like transform
        """
        self.root = root
        self.transform = transform

        df = pd.read_csv(csv_path)
        assert "path" in df.columns and "label" in df.columns, \
            f"CSV must contain columns: path, label. Got: {df.columns.tolist()}"

        # store samples in the same style: [abs_path, int_label]
        self.samples = []
        for p, y in zip(df["path"].tolist(), df["label"].tolist()):
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
        uq_idx = self.uq_idxs[idx]

        # load image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, int(label), int(uq_idx)


def subsample_dataset(dataset, idxs):
    """
    Keep same behavior as sample code:
    - mask samples/targets/uq_idxs
    - ensure int labels
    """
    mask = np.zeros(len(dataset)).astype("bool")
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples, dtype=object)[mask].tolist()
    dataset.targets = np.array(dataset.targets, dtype=object)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(27)):
    """
    Keep instances whose original label in include_classes.
    Then remap labels to 0..len(include_classes)-1
    """
    cls_idxs = [i for i, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {k: i for i, k in enumerate(include_classes)}

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[int(x)]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):
    """
    Balanced val: sample fixed number per class.
    """
    train_classes = list(set(train_dataset.targets))

    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # if a class has fewer images than val_instances_per_class, take all as val
        if len(cls_idxs) <= val_instances_per_class:
            v_ = cls_idxs
            t_ = []
        else:
            v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
            t_ = [x for x in cls_idxs if x not in set(v_)]

        train_idxs.extend(list(t_))
        val_idxs.extend(list(v_))

    return train_idxs, val_idxs


# -----------------------------
# Default known/unknown split (medical view, relaxed B to be larger)
# Known (A类：正常/部位/视野/器械) -> DEFAULT train_classes
# Labels from your table:
# 0 Accessory tools
# 4 Cecum
# 8 Duodenal bulb
# 15 GEJ normal z-line
# 16 Ileocecal valve
# 18 Normal esophagus
# 19 Normal mucosa/vascular pattern (large bowel)
# 20 Normal stomach
# 21 Pylorus
# 24 Retroflex rectum
# 25 Small bowel terminal ileum
DEFAULT_KNOWN_CLASSES = [0, 4, 8, 15, 16, 18, 19, 20, 21, 24, 25]


def get_gastrovision_datasets(
        train_transform,
        test_transform,
        train_classes=DEFAULT_KNOWN_CLASSES,
        prop_train_labels=0.8,
        seed=0,
        split_train_val=False,
        val_instances_per_class=5,
        train_csv_name="gastrovision_label.csv",
        test_csv_name=None
):
    """
    Keep interface/behavior aligned with get_herbarium_datasets.

    - Uses one CSV by default for both train and test (common in GCD setups).
      If you have a separate test CSV, pass test_csv_name.

    Returns dict with keys:
        train_labelled, train_unlabelled, val, test
    """
    np.random.seed(seed)

    train_csv_path = os.path.join(gastrovision_dataroot, train_csv_name)
    if test_csv_name is None:
        test_csv_path = train_csv_path
    else:
        test_csv_path = os.path.join(gastrovision_dataroot, test_csv_name)

    # Init entire "train pool"
    train_dataset = GastrovisionDataset27(
        root=gastrovision_dataroot,
        csv_path=train_csv_path,
        transform=train_transform
    )

    # Labelled set: keep known classes + subsample indices by prop_train_labels
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(
        train_dataset_labelled,
        prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Optionally split labelled into train/val
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(
            train_dataset_labelled,
            val_instances_per_class=val_instances_per_class
        )
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    # Unlabelled set: everything else from the pool (includes unknown classes + unused known instances)
    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    # Test set
    test_dataset = GastrovisionDataset27(
        root=gastrovision_dataroot,
        csv_path=test_csv_path,
        transform=test_transform
    )

    # Build unified target remap dict: known classes first, then remaining classes
    all_classes = sorted(list(set(train_dataset.targets)))
    unlabelled_classes = [c for c in all_classes if c not in list(train_classes)]
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[int(k)] = int(i)

    test_dataset.target_transform = lambda x: target_xform_dict[int(x)]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[int(x)]

    # finalize outputs
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": val_dataset_labelled,
        "test": test_dataset,
    }
    return all_datasets


if __name__ == "__main__":
    # quick sanity check
    np.random.seed(0)

    x = get_gastrovision_datasets(
        train_transform=None,
        test_transform=None,
        train_classes=DEFAULT_KNOWN_CLASSES,
        prop_train_labels=0.5,
        seed=0,
        split_train_val=False
    )

    print("Printing lens...")
    for k, v in x.items():
        if v is not None:
            print(f"{k}: {len(v)}")

    print("Printing labelled and unlabelled overlap...")
    print(set(x["train_labelled"].uq_idxs).intersection(set(x["train_unlabelled"].uq_idxs)))

    print("Printing total instances in train pool...")
    print(len(set(x["train_labelled"].uq_idxs)) + len(set(x["train_unlabelled"].uq_idxs)))

    print(f"Num Labelled Classes: {len(set(x['train_labelled'].targets))}")
    print(f"Num Unlabelled Classes (after remap): {len(set(x['train_unlabelled'].targets))}")
