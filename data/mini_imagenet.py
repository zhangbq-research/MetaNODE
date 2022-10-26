from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt

from PIL import Image

_MINI_IMAGENET_DATASET_DIR = '../datasets/few_shot_data/MiniImagenet'

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

class MiniImageNet(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        self.base_folder = 'miniImagenet'
        assert(phase=='train' or phase=='val' or phase=='test' or phase=='all_train')
        self.phase = phase
        self.name = 'MiniImageNet_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_train.pickle')
        file_train_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_val.pickle')
        file_train_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_test.pickle')
        file_val_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_test.pickle')

        if self.phase=='train':
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                data_base = load_data(file_train_categories_test_phase)
                data_novel = load_data(file_test_categories_test_phase)
            else:
                data_base = load_data(file_train_categories_val_phase)
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate(
                [data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
        elif self.phase=='all_train':
            data_train = load_data(file_train_categories_train_phase)
            data_test = load_data(file_test_categories_test_phase)
            data_val = load_data(file_val_categories_val_phase)

            self.data = np.concatenate(
                [data_train['data'], data_test['data'], data_val['data']], axis=0)
            self.labels = data_train['labels'] + data_test['labels']+ data_val['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5,
                 nKbase=-1,
                 nExemplars=1,
                 nTestNovel=15*5,
                 nTestBase=15*5,
                 batch_size=1,
                 num_workers=4,
                 epoch_size=2000,
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            Kbase = sorted(self.sampleCategories('base', nKbase))
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        Tbase = []
        if len(Kbase) > 0:
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self):
        rand_seed = 45
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self):
        return self.get_iterator()

    def __len__(self):
        return (self.epoch_size / self.batch_size)
