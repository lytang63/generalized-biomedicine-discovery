import sys
import os
sys.path.append("/mnt/d/hanqi/codes/GBD/SelEx")

# import os
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from config import chestxray14_dataroot


class ChestXray14Dataset(Dataset):

    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform

        df = pd.read_csv(csv_path)
        assert "path" in df.columns and "label" in df.columns

        self.samples = []
        for p, y in zip(df["path"].tolist(), df["label"].tolist()):
            abs_path = os.path.join(self.root + "/images", str(p))
            self.samples.append([abs_path, int(y)])

        self.targets = [int(x[1]) for x in self.samples]
        self.uq_idxs = np.arange(len(self))
        self.target_transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        uq_idx = self.uq_idxs[idx]

        img = Image.open(img_path).convert("RGB")
        # img = None

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, int(label), int(uq_idx)


def subsample_dataset(dataset, idxs):
    dataset.samples = np.array(dataset.samples, dtype=object)[idxs].tolist()
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]
    return dataset


def subsample_instances(dataset, prop=0.8, seed=0):
    rng = np.random.RandomState(seed)
    return rng.choice(
        np.arange(len(dataset)),
        replace=False,
        size=int(prop * len(dataset))
    )


def get_chestxray14_datasets(
    train_transform,
    test_transform,
    train_classes,
    prop_train_labels=0.8,
    seed=0,
    split_train_val=False,
    val_instances_per_class=5,
    train_csv_name="chestxray14_label_5percent.csv",
    test_csv_name=None
):

    train_csv_path = os.path.join(chestxray14_dataroot, train_csv_name)
    test_csv_path = train_csv_path if test_csv_name is None \
        else os.path.join(chestxray14_dataroot, test_csv_name)

    # full dataset
    full_dataset = ChestXray14Dataset(
        root=chestxray14_dataroot,
        csv_path=train_csv_path,
        transform=train_transform
    )

    all_indices = np.arange(len(full_dataset))
    targets_array = np.array(full_dataset.targets)

    # known class mask
    known_mask = np.isin(targets_array, train_classes)
    known_indices = np.where(known_mask)[0]

    # subsample known for labelled
    labelled_indices = subsample_instances(
        full_dataset,
        prop=prop_train_labels,
        seed=seed
    )
    labelled_indices = np.intersect1d(labelled_indices, known_indices)

    train_labelled = subsample_dataset(
        deepcopy(full_dataset),
        labelled_indices
    )

    # unlabelled = rest
    unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

    train_unlabelled = subsample_dataset(
        deepcopy(full_dataset),
        unlabelled_indices
    )

    # test dataset
    test_dataset = ChestXray14Dataset(
        root=chestxray14_dataroot,
        csv_path=test_csv_path,
        transform=test_transform
    )

    # unified remap
    all_classes = sorted(list(set(full_dataset.targets)))
    remap_dict = {c: i for i, c in enumerate(all_classes)}

    for d in [train_labelled, train_unlabelled, test_dataset]:
        d.target_transform = lambda x, m=remap_dict: m[int(x)]

    return {
        "train_labelled": train_labelled,
        "train_unlabelled": train_unlabelled,
        "val": None,
        "test": test_dataset
    }


if __name__ == "__main__":

    x = get_chestxray14_datasets(
        train_transform=None,
        test_transform=None,
        train_classes=[2, 4, 5, 6],
        prop_train_labels=0.5,
        seed=0
    )

    print("Train labelled:", len(x["train_labelled"]))
    print("Train unlabelled:", len(x["train_unlabelled"]))
    print("Test:", len(x["test"]))

    # 正确统计 remap 后类别数
    def count_classes(dataset):
        labels = [dataset[i][1] for i in range(len(dataset))]
        return len(set(labels))

    print("Labelled classes:", count_classes(x["train_labelled"]))
    print("Unlabelled classes:", count_classes(x["train_unlabelled"]))

