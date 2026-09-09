from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pets import get_pets_datasets


from data.cifar import subsample_classes as subsample_dataset_cifar
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
from data.pets import subsample_classes as subsample_dataset_pets


from data.GBD.gastrovision import get_gastrovision_datasets
from data.GBD.chestxray14 import get_chestxray14_datasets
from data.GBD.fitzpatrick17k import get_fitzpatrick17k_datasets
from data.GBD.histoset import get_histoset_datasets
from data.GBD.mll23 import get_mll23_datasets
from data.GBD.derm12345 import get_derm12345_datasets

from data.GBD.gastrovision import subsample_classes as subsample_dataset_gastro


from copy import deepcopy
import pickle
import os

from config import osr_split_dir

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'pets': subsample_dataset_pets,
'gastrovision': subsample_dataset_gastro,
# 'chestxray14': subsample_dataset_gastro,
# 'fitzpatrick17k': subsample_dataset_gastro

}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'pets': get_pets_datasets,
'gastrovision': get_gastrovision_datasets,
'chestxray14': get_chestxray14_datasets,
'fitzpatrick17k': get_fitzpatrick17k_datasets,
'histoset': get_histoset_datasets,
'mll23': get_mll23_datasets,
'derm12345': get_derm12345_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    # 
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']


    # GBD: Biomedical datasets
    elif args.dataset_name == 'gastrovision':

        args.image_size = 224

        if args.mode == 1:
            # Mode 1: Semantic heuristic (normal/anatomical as known)
            args.train_classes = [0, 4, 8, 15, 16, 18, 19, 20, 21, 24, 25]
        if args.mode == 2:
            # Mode 2: Long-tail distribution (top 50% frequent classes as known)
            args.train_classes = [19, 0, 20, 25, 6, 21, 15, 10, 8, 16, 3, 9, 18, 7]
        if args.mode == 3:
            # Mode 3: Hierarchical split
            args.train_classes = [20, 15, 2, 13, 14, 19, 4, 5, 1, 17, 9, 10, 3, 0]

        args.unlabeled_classes = [i for i in range(27) if i not in args.train_classes]
    
    elif args.dataset_name == 'derm12345':
        args.image_size = 224
        if args.mode == 1:
            # head classes as known
            args.train_classes = [22, 23, 11, 17, 2,  8,  12, 3,  36, 37, 18, 20, 31, 16, 30, 27, 7,  5,  38, 4]
        elif args.mode == 2:
            # tail classes as known
            print("mode 2 be selected; dataset: derm12345!")
            args.train_classes = [23, 22, 11, 37, 17, 14, 2, 8, 12, 3, 36, 18, 20, 10, 31, 16, 30, 27, 7, 38]
        elif args.mode == 3:
            # balanced split
            print("mode 3 be selected; dataset: derm12345!")
            args.train_classes = [0, 12, 10, 3, 16, 1, 13, 3, 39, 34, 21, 29, 7, 5, 27, 37, 18, 6, 20, 25, 4, 8, 9, 15, 19, 24]
        args.unlabeled_classes = [i for i in range(40) if i not in args.train_classes]

    elif args.dataset_name == 'fitzpatrick17k':
        args.image_size = 224
        if args.mode == 1:
            # head classes as known
            args.train_classes = [4, 7, 3, 0]
        elif args.mode == 2:
            # tail classes as known
            args.train_classes = [2, 5, 6, 8]
        elif args.mode == 3:
            # balanced split
            args.train_classes = [0, 2, 4, 6, 8]
        args.unlabeled_classes = [i for i in range(9) if i not in args.train_classes]
    
    elif args.dataset_name == 'chestxray14':
        args.image_size = 224
        if args.mode == 1:
            # head classes
            args.train_classes = [10, 8, 0, 4, 11, 9, 14]
        elif args.mode == 2:
            # tail classes
            args.train_classes = [13, 7, 3, 6, 5, 1, 12]
        elif args.mode == 3:
            # balanced split
            args.train_classes = [0, 2, 4, 6, 8, 10, 12]
        args.unlabeled_classes = [i for i in range(15) if i not in args.train_classes]

    elif args.dataset_name == 'histoset':

        args.image_size = 224

        # Mode 1: Semantic heuristic
        # Mode 2: Long-tail distribution (top 50%)
        # Mode 3: Hierarchical split

        mode = getattr(args, "mode", 1)
        if mode == 1:
            args.train_classes = [0, 3, 5, 7, 12]
        elif mode == 2:
            args.train_classes = [0, 1, 2, 3, 4, 5, 6]
        elif mode == 3:
            args.train_classes = [0, 2, 4, 5, 7, 9, 10, 11]
        else:
            raise ValueError(f"Unsupported mode={mode} for histoset. Use 1/2/3.")

        args.unlabeled_classes = [i for i in range(14) if i not in args.train_classes]
    
    elif args.dataset_name == 'mll23':

        args.image_size = 224

        # 18 classes in total
        mode = getattr(args, "mode", 1)

        if mode == 1:
            # Mode 1: Semantic heuristic (normal cell types as known)
            args.train_classes = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        elif mode == 2:
            # Mode 2: Long-tail distribution (top 50% by frequency)
            args.train_classes = [12, 9, 3, 2, 8, 1, 13, 16, 4]

        elif mode == 3:
            # Mode 3: Hierarchical split based on cell lineage taxonomy
            args.train_classes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 13]

        else:
            raise ValueError(f"Unsupported mode={mode} for mll23. Use 1/2/3.")

        args.unlabeled_classes = [i for i in range(18) if i not in args.train_classes]

    elif args.dataset_name == 'imagenet_100':
        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']


        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'pets':

        args.image_size = 224
        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)

    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    else:

        raise NotImplementedError

    return args
