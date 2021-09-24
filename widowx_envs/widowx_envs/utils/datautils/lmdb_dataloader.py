import os
from cv2 import data
from widowx_envs.utils.utils import AttrDict
import random
import cv2
from PIL import Image
import sys
import six
import string
import argparse
import numpy as np
import pandas as pd
import json

import lmdb
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
import pyarrow as pa

from widowx_envs.utils.datautils.data_augmentation import get_random_crop, get_random_color_aug
from widowx_envs.utils.datautils.base_data_loader import BaseVideoDataset

from pathlib import Path
import glob


def print_dataset_stats(dataframe):
    dataset_stats = ''
    dataset_stats += 'number of images {} \n'.format(len(dataframe))
    tasks = dataframe.policy_desc.unique()
    for task in tasks:
        dataset_stats += '#######################################################\n'
        dataset_stats += 'task {} \n'.format(task)
        dataset_stats += 'number of trajs {} \n'.format(len(dataframe[(dataframe['policy_desc'] == task) & (dataframe['tstep'] == 0) & (dataframe['camera_index'] == 0)]))
        dataset_stats += 'cams {} \n'.format(dataframe[(dataframe['policy_desc'] == task)].camera_index.unique())
        dataset_stats += 'number of camera variations {} \n'.format(len(dataframe[(dataframe['policy_desc'] == task)].camera_variation.unique()))
        dataset_stats += 'number of domain hashes {} \n'.format(len(dataframe[(dataframe['policy_desc'] == task)].domain_hash.unique()))
        dataset_stats += 'number of backgrounds {} \n'.format(len(dataframe[(dataframe['policy_desc'] == task)].background.unique()))
        dataset_stats += 'backgrounds {} \n'.format(dataframe[(dataframe['policy_desc'] == task)].background.unique())
        if hasattr(dataframe, 'environment'):
            dataset_stats += 'number of environments {} \n'.format(len(dataframe[(dataframe['policy_desc'] == task)].environment.unique()))
            dataset_stats += 'environments {} \n'.format(dataframe[(dataframe['policy_desc'] == task)].environment.unique())


    dataset_stats += '#######################################################\n'
    dataset_stats += '-------------------------------------------------------\n'
    dataset_stats += 'overall number of trajs {} \n'.format(len(dataframe[(dataframe['tstep'] == 0) & (dataframe['camera_index'] == 0)]))
    dataset_stats += 'overall number of tasks {} \n'.format(len(tasks))
    dataset_stats += 'overall number of domain_hashes {} \n'.format(len(dataframe.domain_hash.unique()))
    dataset_stats += 'overall number of object combinations {} \n'.format(len(dataframe.object_classes.unique()))
    if hasattr(dataframe, 'environment'):
        dataset_stats += 'overall environments {} \n'.format(dataframe.environment.unique())

    print(dataset_stats)
    return dataset_stats


def get_dirs_recursive(dir, excluded=None):
    print('globbing for lmdb folders in ', dir)
    lmdb_paths_depth0 = glob.glob(dir + '/lmdb', recursive=True)
    lmdb_paths_depth1 = glob.glob(dir + '/*/lmdb', recursive=True)
    lmdb_paths_depth2 = glob.glob(dir + '/*/*/lmdb', recursive=True)

    lmdb_paths = lmdb_paths_depth0 + lmdb_paths_depth1 + lmdb_paths_depth2

    if excluded is None:
        return lmdb_paths
    else:
        new_lmdb_paths = []
        for d in lmdb_paths:
            reject = False
            for exdir in excluded:
                if exdir in d:
                    reject = True
                    break
            if not reject:
                new_lmdb_paths.append(d)
        return new_lmdb_paths


class LMDB_Dataset(BaseVideoDataset):
    def __init__(self, data_conf, phase='train', shuffle=True):

        BaseVideoDataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self.phase = phase
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        if not isinstance(self._hp.data_dir, list):
            if 'lmdb' not in self._hp.data_dir:
                self._hp.data_dir = get_dirs_recursive(self._hp.data_dir)
            else:
                self._hp.data_dir = [self._hp.data_dir]

        self.databases_dict = {}
        self.global_indices = []

        domain_hashes = set()
        task_descriptions = set()

        for i, database in enumerate(self._hp.data_dir):
            override_selcamera = None
            if isinstance(database, dict):
                database_dir = database.dir
                override_selcamera = database.sel_camera
            else:
                database_dir = database

            database_i = AttrDict()
            database_i.data_base = lmdb.open(os.path.join(database_dir, phase, 'lmdb.data'), subdir=False,
                                      readonly=True, lock=False,
                                      readahead=False, meminit=False)
            with database_i.data_base.begin(write=False) as txn:
                # self.length = pa.deserialize(txn.get(b'__len__')) - 1

                database_i.meta_data_list = pa.deserialize(txn.get(b'meta_data_list'))
                database_i.index_dict = pa.deserialize(txn.get(b'index_dict'))
                database_i.lmdb_keys = pa.deserialize(txn.get(b'__keys__'))

            self.databases_dict['database{}'.format(i)] = database_i

            print('loading database {} with {} trajectories'.format(database_dir, len(database_i.index_dict)))
            # concatenating databases and creating global index

            for traj in database_i.index_dict.keys():
                for t in database_i.index_dict[traj]:
                    if override_selcamera is not None:
                        cam_indices = ['cam{}'.format(override_selcamera)]
                    elif self._hp.sel_camera == 'random':
                        cam_indices = database_i.index_dict[traj][t]
                    else:
                        cam_indices = ['cam{}'.format(self._hp.sel_camera)]
                    for cam_ind in cam_indices:
                        self.global_indices.append(['database{}'.format(i), (traj, t, cam_ind)])

                        list_index = database_i.index_dict[traj][t][cam_ind]
                        meta_data_dict = database_i.meta_data_list[list_index]
                        domain_hashes.add(meta_data_dict['domain_hash'])
                        task_descriptions.add(meta_data_dict['policy_desc'])

        if phase == 'train':
            self.domain_hash_index = {hash: index for hash, index in
                                                     zip(domain_hashes, range(len(domain_hashes)))}
            print('found {} domain hashes'.format(len(domain_hashes)))

            self.taskdescription2task_index = {task_descp: index for task_descp, index in
                                                     zip(task_descriptions, range(len(task_descriptions)))}
            print('found {} task descriptions'.format(len(task_descriptions)))
            print(self.taskdescription2task_index)

        random.shuffle(self.global_indices)
        self.global_indices = {i: traj for i, traj in zip(range(len(self.global_indices)), self.global_indices)}

    def set_domain_and_taskdescription_indices(self, domain_index, task_index):
        """
        This is to make sure that the train and val dataloaders are using the same domain_has_index and taskdescription2task_index
        """
        assert self.phase == 'val' or self.phase == 'test'
        self.domain_hash_index = domain_index
        self.taskdescription2task_index = task_index

    def get_num_domains(self):
        return len(self.domain_hash_index.keys())

    def _default_hparams(self):
        default_dict = AttrDict(
            name='',
            color_augmentation=False,
            random_crop=False,
            image_size_beforecrop=None,
            data_dir=None,
            sel_camera='random',  # -1 means selecting all cameras,
            normalize_images=False,
            sel_len=-1,  # number of time steps for contigous sequence that is shifted within sequeence of T randomly
            get_final_image=True,
        )
        parent_params = AttrDict(super()._default_hparams())
        parent_params.update(default_dict)
        return parent_params

    def __getitem__(self, index):
        # making sure that different loading threads aren't using the same random seed.
        np.random.seed(index)
        random.seed(index)

        selected_database, [i_traj, t, cam_ind] = self.global_indices[index]

        index_dict = self.databases_dict[selected_database].index_dict
        lmdb_keys = self.databases_dict[selected_database].lmdb_keys
        database = self.databases_dict[selected_database].data_base

        ncam = len(index_dict[i_traj]['t0'].keys())
        tlen = len(index_dict[i_traj].keys())

        lmdb_index = index_dict[i_traj][t][cam_ind]
        lmdb_index_finalstep = index_dict[i_traj]['t{}'.format(tlen - 1)][cam_ind]

        with database.begin(write=False) as txn:
            byteflow = txn.get(lmdb_keys[lmdb_index])
            current_tstep = pa.deserialize(byteflow)

            if self._hp.get_final_image:
                byteflow = txn.get(lmdb_keys[lmdb_index_finalstep])
                final_tstep = pa.deserialize(byteflow)

        image = cv2.imdecode(current_tstep['env_obs']['image'], cv2.IMREAD_COLOR)

        meta_data = current_tstep['meta_data']

        data_dict = AttrDict(
            images=self._process_images(image),
            actions=current_tstep['policy_out']['actions'].astype(np.float32),
            states=current_tstep['env_obs']['state'].astype(np.float32),
            task_id=self.taskdescription2task_index[meta_data['policy_desc']]
        )

        data_dict['domain_ind'] = self.domain_hash_index[meta_data['domain_hash']]

        if self._hp.get_final_image:
            final_image = cv2.imdecode(final_tstep['env_obs']['image'], cv2.IMREAD_COLOR)
            data_dict['final_image'] = self._process_images(final_image)

        return data_dict

    def __len__(self):
        return len(self.global_indices.keys())

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self._hp.name + ')'

    def _process_images(self, images):

        if images.shape[-2:] != self._hp.image_size_beforecrop:
            target_height, target_width= self._hp.image_size_beforecrop
            images = cv2.resize(images, (target_width, target_height), interpolation=cv2.INTER_AREA)[:, :, ::-1]        # cast and normalize images (if set)
        if len(images.shape) == 3:
            images = np.transpose(images, (2, 0, 1)).astype(np.float32) / 255
        if self._hp['normalize_images']:
            images -= self._mean
            images /= self._std

        images = self.apply_data_augmentation(images)
        images = images * 2 - 1
        return images

    def apply_data_augmentation(self, images):
        if self._hp.random_crop:
            if self.phase == 'train':
                images = get_random_crop(images, self._hp.random_crop)
            else:
                images = get_random_crop(images, self._hp.random_crop, center_crop=True)
        if self._hp.color_augmentation and self.phase == 'train':
            images = get_random_color_aug(images, self._hp.color_augmentation)
        return images
        

class LMDB_Dataset_Pandas(BaseVideoDataset):
    def __init__(self, data_conf, phase='train', shuffle=True):

        BaseVideoDataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self.phase = phase
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        if not isinstance(self._hp.data_dir, list):
            if 'lmdb' != self._hp.data_dir[-4:]:
                self._hp.data_dir = get_dirs_recursive(self._hp.data_dir, self._hp.excluded_dirs)
            else:
                self._hp.data_dir = [self._hp.data_dir]
            print('found {} lmdb directories.'.format(len(self._hp.data_dir)))

        self.databases_dict = {}
        self.list_meta_data_dict = []

        for i, database in enumerate(self._hp.data_dir):
            if isinstance(database, dict):
                database_dir = database.dir
            else:
                database_dir = database

            database_i = AttrDict()
            database_i.data_base = lmdb.open(os.path.join(database_dir, phase, 'lmdb.data'), subdir=False,
                                      readonly=True, lock=False,
                                      readahead=False, meminit=False)
            with database_i.data_base.begin(write=False) as txn:

                database_i.meta_data_list = pa.deserialize(txn.get(b'meta_data_list'))
                database_i.index_dict = pa.deserialize(txn.get(b'index_dict'))
                database_i.lmdb_keys = pa.deserialize(txn.get(b'__keys__'))

            self.databases_dict[database_dir] = database_i

            print('opened database {} with {} trajectories'.format(database_dir, len(database_i.index_dict)))
            # concatenating databases and creating global index

            for traj in database_i.index_dict.keys():
                for t in database_i.index_dict[traj]:
                    for cam_ind in database_i.index_dict[traj][t]:

                        list_index = database_i.index_dict[traj][t][cam_ind]
                        meta_data_dict = database_i.meta_data_list[list_index]
                        date = meta_data_dict['camera_variation'].split('_')[0]
                        month, day = date.split('-')[0], date.split('-')[1]
                        tstep_reverse = meta_data_dict['term_t'] - meta_data_dict['tstep']
                        database_location_dict = {'database_dir': database_dir, 'traj':traj, 'tstep_reverse': tstep_reverse, 'month': int(month), 'day': int(day)}
                        meta_data_dict.update(database_location_dict)
                        meta_data_dict['object_classes'] = ','.join(meta_data_dict['object_classes'])

                        if 'policy_type' not in meta_data_dict.keys():
                            meta_data_dict['policy_type'] = 'None'

                        self.list_meta_data_dict.append(meta_data_dict)


        self.dataframe = pd.DataFrame(data=self.list_meta_data_dict)

        if self._hp.filtering_function:
            # Apply the filter conditions
            for func in self._hp.filtering_function:
                print('before filtering cameras', self.dataframe.camera_index.unique())
                self.dataframe = func(self.dataframe)
            
        # Filter out timesteps without human intervention
        if self._hp.filter_human_intervention:
            print("before filtering human intervention", len(self.dataframe))
            self.dataframe = self.dataframe[(self.dataframe['policy_type'] != 'GCBCPolicyImages')]
            print("after filtering human intervention", len(self.dataframe))

        print('dataset {} phase: {} uses aliasing dict {}'.format(self._hp.name, self.phase, self._hp.aliasing_dict != None))
        if self._hp.aliasing_dict:
            for key, value in self._hp.aliasing_dict.items():
                self.dataframe.loc[self.dataframe['policy_desc']==key, 'policy_desc'] = value
        
        domain_hashes = self.dataframe.domain_hash.unique()
        self.domain_hash_index = {hash: index for hash, index in zip(domain_hashes, range(len(domain_hashes)))}
        print('found {} domain hashes'.format(len(domain_hashes)))
        task_descriptions = self.dataframe.policy_desc.unique()

        self.taskdescription2task_index = {task_descp: index for task_descp, index in
                                            zip(task_descriptions, range(len(task_descriptions)))}
        print('found {} task descriptions'.format(len(task_descriptions)))
        print(self.taskdescription2task_index)

        # Shuffle the filtered dataframe
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        assert (not self._hp.get_final_image) or (self._hp.get_final_image_from_same_traj)
        
        self.dataset_stats = print_dataset_stats(self.dataframe)
        self.dataset_stats = self._hp.name + "\n" + self.dataset_stats

    def set_domain_and_taskdescription_indices(self, domain_index, task_index):
        """
        This is to make sure that the train and val dataloaders are using the same domain_has_index and taskdescription2task_index
        """
        self.domain_hash_index = domain_index
        self.taskdescription2task_index = task_index

    def get_num_domains(self):
        return len(self.domain_hash_index.keys())

    def _default_hparams(self):
        default_dict = AttrDict(
            name='',
            color_augmentation=False,
            random_crop=False,
            image_size_beforecrop=None,
            data_dir=None,
            excluded_dirs=[],
            normalize_images=False,
            concat_random_cam=False,
            get_final_image=False,
            get_next_image=False,
            filtering_function=None,
            get_final_image_from_same_traj=False,
            final_image_match=None,
            stack_goal_images=1,
            aliasing_dict=None,
            filter_human_intervention=True,
        )
        parent_params = AttrDict(super()._default_hparams())
        parent_params.update(default_dict)
        return parent_params

    def __getitem__(self, index):
        # making sure that different loading threads aren't using the same random seed.
        np.random.seed(index)
        random.seed(index)

        datapoint = self.dataframe.loc[index]
        database_dir = datapoint['database_dir']
        tstep = datapoint['tstep']
        camera_index = datapoint['camera_index']
        i_traj = datapoint['traj']
        policy_desc =  datapoint['policy_desc']
        domain_hash = datapoint['domain_hash']
        object_classes = datapoint['object_classes']
        tstep_reverse = datapoint['tstep_reverse']

        cam_ind = 'cam{}'.format(camera_index)
        t = 't{}'.format(tstep)

        index_dict = self.databases_dict[database_dir].index_dict
        lmdb_keys = self.databases_dict[database_dir].lmdb_keys
        database = self.databases_dict[database_dir].data_base

        with database.begin(write=False) as txn:

            lmdb_index = index_dict[i_traj][t][cam_ind]
            byteflow = txn.get(lmdb_keys[lmdb_index])
            current_tstep = pa.deserialize(byteflow)

            if self._hp.get_final_image and self._hp.get_final_image_from_same_traj:
                tlen = len(index_dict[i_traj].keys())
                lmdb_index_finalstep = index_dict[i_traj]['t{}'.format(tlen - 1)][cam_ind]
                byteflow = txn.get(lmdb_keys[lmdb_index_finalstep])
                final_tstep = pa.deserialize(byteflow)
            
            if self._hp.get_next_image:
                lmdb_index_next = index_dict[i_traj]['t{}'.format(tstep + 1)][cam_ind]
                byteflow = txn.get(lmdb_keys[lmdb_index_next])
                next_tstep = pa.deserialize(byteflow)

            if self._hp.concat_random_cam:
                num_cam = len(index_dict[i_traj]['t{}'.format(tstep)].keys())
                assert num_cam > 0
                if num_cam == 1:
                    cam_ind = 0
                else:
                    cam_ind = np.random.randint(1, num_cam)
                lmdb_index_other_cam = index_dict[i_traj]['t{}'.format(tstep)]['cam{}'.format(cam_ind)]
                byteflow = txn.get(lmdb_keys[lmdb_index_other_cam])
                random_other_cam = pa.deserialize(byteflow)
                random_other_cam = self._process_images(cv2.imdecode(random_other_cam['env_obs']['image'], cv2.IMREAD_COLOR))

        image = self._process_images(cv2.imdecode(current_tstep['env_obs']['image'], cv2.IMREAD_COLOR))

        meta_data = current_tstep['meta_data']
        policy_out = current_tstep['policy_out']
        if 'policy_desc' in policy_out.keys():
            meta_data['policy_desc'] = policy_out['policy_desc']

        if self._hp.aliasing_dict is not None:
            if meta_data['policy_desc'] in self._hp.aliasing_dict.keys():
                meta_data['policy_desc'] = self._hp.aliasing_dict[meta_data['policy_desc']]

        data_dict = AttrDict(
            actions=current_tstep['policy_out']['actions'].astype(np.float32),
            states=current_tstep['env_obs']['state'].astype(np.float32),
            rewards=np.array([tstep_reverse < 2]).astype(np.float32),
            terminals=np.array([tstep_reverse == 1]).astype(np.int),
            task_id=self.taskdescription2task_index[meta_data['policy_desc']]
        )

        if self._hp.concat_random_cam:
            data_dict.images = np.concatenate([image, random_other_cam], axis=0)
        else:
            data_dict.images = image

        data_dict['domain_ind'] = self.domain_hash_index[meta_data['domain_hash']]

        if self._hp.get_next_image:
            next_image = cv2.imdecode(next_tstep['env_obs']['image'], cv2.IMREAD_COLOR)
            data_dict['next_images'] = [self._process_images(next_image)]

        if self._hp.get_final_image and self._hp.get_final_image_from_same_traj:
            final_image = cv2.imdecode(final_tstep['env_obs']['image'], cv2.IMREAD_COLOR)
            data_dict['final_image'] = [self._process_images(final_image)]
            final_step_meta_data = final_tstep['meta_data']
            data_dict['final_image_domain_ind'] = self.domain_hash_index[final_step_meta_data['domain_hash']]
        else:
            data_dict['final_image'] = data_dict['images']

        return data_dict

    def __len__(self):
        # print("dataset length", len(self.dataframe.index))
        return len(self.dataframe.index)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self._hp.name + ')'

    def _process_images(self, images):

        if images.shape[-2:] != self._hp.image_size_beforecrop:
            target_height, target_width= self._hp.image_size_beforecrop
            images = cv2.resize(images, (target_width, target_height), interpolation=cv2.INTER_AREA)[:, :, ::-1]        # cast and normalize images (if set)
        if len(images.shape) == 3:
            images = np.transpose(images, (2, 0, 1)).astype(np.float32) / 255
        if self._hp['normalize_images']:
            images -= self._mean
            images /= self._std

        images = self.apply_data_augmentation(images)
        images = images * 2 - 1
        return images

    def apply_data_augmentation(self, images):
        if self._hp.random_crop:
            if self.phase == 'train':
                images = get_random_crop(images, self._hp.random_crop)
            else:
                images = get_random_crop(images, self._hp.random_crop, center_crop=True)
        if self._hp.color_augmentation and self.phase == 'train':
            images = get_random_color_aug(images, self._hp.color_augmentation)
        return images

class LMDB_Dataset_Success_Classifier(BaseVideoDataset):
    def __init__(self, data_conf, phase='train', shuffle=True):

        BaseVideoDataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self.phase = phase
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        if not isinstance(self._hp.data_dir, list):
            if 'lmdb' != self._hp.data_dir[-4:]:
                self._hp.data_dir = get_dirs_recursive(self._hp.data_dir, self._hp.excluded_dirs)
            else:
                self._hp.data_dir = [self._hp.data_dir]
            print('found {} lmdb directories.'.format(len(self._hp.data_dir)))

        self.databases_dict = {}
        self.list_meta_data_dict = []

        for i, database in enumerate(self._hp.data_dir):
            if isinstance(database, dict):
                database_dir = database.dir
            else:
                database_dir = database

            print('opening database', database)
            database_i = AttrDict()
            database_i.data_base = lmdb.open(os.path.join(database_dir, phase, 'lmdb.data'), subdir=False,
                                      readonly=True, lock=False,
                                      readahead=False, meminit=False)
            with database_i.data_base.begin(write=False) as txn:

                database_i.meta_data_list = pa.deserialize(txn.get(b'meta_data_list'))
                database_i.index_dict = pa.deserialize(txn.get(b'index_dict'))
                database_i.lmdb_keys = pa.deserialize(txn.get(b'__keys__'))

            self.databases_dict[database_dir] = database_i

            print('opened database {} with {} trajectories'.format(database_dir, len(database_i.index_dict)))
            # concatenating databases and creating global index

            for traj in database_i.index_dict.keys():
                for t in database_i.index_dict[traj]:
                    for cam_ind in database_i.index_dict[traj][t]:
                        list_index = database_i.index_dict[traj][t][cam_ind]
                        meta_data_dict = database_i.meta_data_list[list_index]
                        tstep_reverse = meta_data_dict['term_t'] - meta_data_dict['tstep']
                        database_location_dict = {'database_dir': database_dir, 'traj':traj, 'tstep_reverse': tstep_reverse}
                        meta_data_dict.update(database_location_dict)
                        meta_data_dict['object_classes'] = ','.join(meta_data_dict['object_classes'])
                        self.list_meta_data_dict.append(meta_data_dict)

        self.dataframe = pd.DataFrame(data=self.list_meta_data_dict)

        if self._hp.filtering_function:
            # Apply the filter conditions
            for func in self._hp.filtering_function:
                self.dataframe = func(self.dataframe)

        if self._hp.aliasing_dict:
            for key, value in self._hp.aliasing_dict.items():
                self.dataframe.loc[self.dataframe['policy_desc']==key, 'policy_desc'] = value
        
        domain_hashes = self.dataframe.domain_hash.unique()
        self.domain_hash_index = {hash: index for hash, index in zip(domain_hashes, range(len(domain_hashes)))}
        print('found {} domain hashes'.format(len(domain_hashes)))
        task_descriptions = self.dataframe.policy_desc.unique()

        self.taskdescription2task_index = {task_descp: index for task_descp, index in
                                            zip(task_descriptions, range(len(task_descriptions)))}
        print('found {} task descriptions'.format(len(task_descriptions)))
        print(self.taskdescription2task_index)

        # Shuffle the filtered dataframe
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataset_stats = print_dataset_stats(self.dataframe)
        self.dataset_stats = self._hp.name + "\n" + self.dataset_stats

    def set_domain_and_taskdescription_indices(self, domain_index, task_index):
        """
        This is to make sure that the train and val dataloaders are using the same domain_has_index and taskdescription2task_index
        """
        self.domain_hash_index = domain_index
        self.taskdescription2task_index = task_index

    def get_num_domains(self):
        return len(self.domain_hash_index.keys())

    def _default_hparams(self):
        default_dict = AttrDict(
            name='',
            color_augmentation=False,
            random_crop=False,
            image_size_beforecrop=None,
            data_dir=None,
            excluded_dirs=[],
            normalize_images=False,
            get_final_image=True,
            filtering_function=None,
            aliasing_dict=None,
        )
        parent_params = AttrDict(super()._default_hparams())
        parent_params.update(default_dict)
        return parent_params

    def __getitem__(self, index):
        # making sure that different loading threads aren't using the same random seed.
        np.random.seed(index)
        random.seed(index)

        datapoint = self.dataframe.loc[index]
        database_dir = datapoint['database_dir']
        tstep = datapoint['tstep']
        tstep_reverse = datapoint['tstep_reverse']
        camera_index = datapoint['camera_index']
        i_traj = datapoint['traj']

        cam_ind = 'cam{}'.format(camera_index)
        t = 't{}'.format(tstep)

        index_dict = self.databases_dict[database_dir].index_dict
        lmdb_keys = self.databases_dict[database_dir].lmdb_keys
        database = self.databases_dict[database_dir].data_base

        with database.begin(write=False) as txn:
            lmdb_index = index_dict[i_traj][t][cam_ind]
            byteflow = txn.get(lmdb_keys[lmdb_index])
            current_tstep = pa.deserialize(byteflow)
        
        image = cv2.imdecode(current_tstep['env_obs']['image'], cv2.IMREAD_COLOR)
        meta_data = current_tstep['meta_data']

        if meta_data['policy_desc'] in self._hp.aliasing_dict.keys():
            meta_data['policy_desc'] = self._hp.aliasing_dict[meta_data['policy_desc']]

        data_dict = AttrDict(
            images=self._process_images(image),
            classes=np.array([tstep_reverse < 2]).astype(np.int),
            states=current_tstep['env_obs']['state'].astype(np.float32),
            task_id=self.taskdescription2task_index[meta_data['policy_desc']]
        )

        data_dict['domain_ind'] = self.domain_hash_index[meta_data['domain_hash']]

        return data_dict

    def __len__(self):
        return len(self.dataframe.index)

    def _process_images(self, images):
        if images.shape[-2:] != self._hp.image_size_beforecrop:
            target_height, target_width= self._hp.image_size_beforecrop
            images = cv2.resize(images, (target_width, target_height), interpolation=cv2.INTER_AREA)[:, :, ::-1]        # cast and normalize images (if set)
        if len(images.shape) == 3:
            images = np.transpose(images, (2, 0, 1)).astype(np.float32) / 255
        if self._hp['normalize_images']:
            images -= self._mean
            images /= self._std

        images = self.apply_data_augmentation(images)
        images = images * 2 - 1
        return images

    def apply_data_augmentation(self, images):
        if self._hp.random_crop:
            if self.phase == 'train':
                images = get_random_crop(images, self._hp.random_crop)
            else:
                images = get_random_crop(images, self._hp.random_crop, center_crop=True)
        if self._hp.color_augmentation and self.phase == 'train':
            images = get_random_color_aug(images, self._hp.color_augmentation)
        return images

class TaskConditioningLMDB_Dataset(LMDB_Dataset):
    def __init__(self, data_conf, phase='train', shuffle=True):
        LMDB_Dataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

    def _default_hparams(self):
        default_dict = AttrDict(
            conditioning_task=''
        )
        parent_params = AttrDict(super()._default_hparams())
        parent_params.update(default_dict)
        return parent_params

    def __getitem__(self, index):
        data_dict = super(TaskConditioningLMDB_Dataset, self).__getitem__(index)
        if data_dict.task_id == self.taskdescription2task_index[self._hp.conditioning_task]:
            data_dict.final_image = np.zeros_like(data_dict.final_image)
        return data_dict

class FinalImageZerosLMDB_Dataset(LMDB_Dataset):
    def __init__(self, data_conf, phase='train', shuffle=True):
        LMDB_Dataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

    def __getitem__(self, index):
        data_dict = super(FinalImageZerosLMDB_Dataset, self).__getitem__(index)
        data_dict.final_image = np.zeros_like(data_dict.final_image)
        return data_dict

class LMDB_Dataset_Goal(LMDB_Dataset_Pandas):
    def __init__(self, data_conf, phase='train', shuffle=True):
        if not hasattr(data_conf, 'filtering_function'):
            data_conf.filtering_function = [filtering_goal]
        else:
            data_conf.filtering_function.append(filtering_goal)
        LMDB_Dataset_Pandas.__init__(self, data_conf, phase=phase, shuffle=shuffle)
        print_dataset_stats(self.dataframe)

    def __getitem__(self, index):
        np.random.seed(index)
        random.seed(index)
        
        goal_images = []

        if len(self.dataframe.index) > self._hp.stack_goal_images:
            sampled_indices = random.sample(list(self.dataframe.index), self._hp.stack_goal_images)
        else:
            sampled_indices = random.sample(list(self.dataframe.index) * self._hp.stack_goal_images, self._hp.stack_goal_images)
        
        for sample_index in sampled_indices:
            datapoint = self.dataframe.loc[sample_index]
            database_dir, tstep, camera_index, i_traj = \
            datapoint['database_dir'], datapoint['tstep'], datapoint['camera_index'], datapoint['traj']
            database = self.databases_dict[database_dir].data_base
            with database.begin(write=False) as txn:
                cam_ind = 'cam{}'.format(camera_index)
                t = 't{}'.format(tstep)
                index_dict = self.databases_dict[database_dir].index_dict
                lmdb_keys = self.databases_dict[database_dir].lmdb_keys

                lmdb_index = index_dict[i_traj][t][cam_ind]
                byteflow = txn.get(lmdb_keys[lmdb_index])
                final_tstep = pa.deserialize(byteflow)

            final_image = cv2.imdecode(final_tstep['env_obs']['image'], cv2.IMREAD_COLOR) 
            goal_images.append(self._process_images(final_image))

        return goal_images

def count_hashes(loader):
    global counter, images, b
    tstart = time.time()
    n_batch = 200
    hashes = set()
    counter = 0
    while True:
        for i_batch, sample_batched in enumerate(loader):
            # print('ibatch', counter)
            images = np.asarray(sample_batched['images'])

            print('domain ind', sample_batched['domain_ind'])
            for b in range(images.shape[0]):
                image_string = images[b].tostring()
                hashes.add(hashlib.sha256(image_string).hexdigest())
            counter += 1

            if counter == n_batch:
                break
        if counter == n_batch:
            break
    print('num hashes ', len(hashes))
    print('average loading time', (time.time() - tstart) / n_batch)

def measure_time(loader):
    tstart = time.time()
    n_batch = 20
    for i_batch, sample_batched in enumerate(loader):
        print('ibatch', i_batch)
        if i_batch == n_batch:
            break
    print('average loading time', (time.time() - tstart)/n_batch)

def save_images(loader):
    for i_batch, sample_batched in enumerate(loader):
        # print('ibatch', i_batch)
        images = sample_batched['images'].numpy()
        save_image_batch(i_batch, images)
        if i_batch == 10:
            break

def save_goal_images(loader):
    for i_batch, sample_batched in enumerate(loader):
        # print('ibatch', i_batch)
        images = sample_batched[0].numpy()
        save_image_batch(i_batch, images)
        if i_batch == 10:
            break

def save_image_and_goal_images(loader):
    for i_batch, sample_batched in enumerate(loader):
        # print('ibatch', i_batch)
        images = sample_batched['images'].numpy()
        goal_images = sample_batched['final_image'].numpy()
        save_image_batch(i_batch, images)
        save_image_batch(i_batch + 11, goal_images)
        if i_batch == 10:
            break

def save_image_batch(i_batch, images):
    images = (np.transpose((images + 1) / 2, [0, 2, 3, 1]) * 255.).astype(
        np.uint8)
    img = np.concatenate(np.split(images, images.shape[0], 0), 2).squeeze()
    Image.fromarray(img).save(os.environ['EXP'] + '/im_b{}.jpg'.format(i_batch))

def save_images_per_domain_ind(dataset, loader):
    per_domain_images = [[] for _ in range(dataset.get_num_domains())]

    per_domain_length = 12
    for i_batch, sample_batched in enumerate(loader):
        # print('ibatch', counter)
        images = np.asarray(sample_batched['images'])
        domain_ind = sample_batched['domain_ind']

        import copy

        for b in range(images.shape[0]):
            domain = domain_ind[b]
            if len(per_domain_images[domain]) < per_domain_length:
                per_domain_images[domain].append(copy.deepcopy(images[b]))

        min_length = min([len(d) for d in per_domain_images])
        print([len(d) for d in per_domain_images])
        if min_length == per_domain_length:
            break

    for i, d in enumerate(per_domain_images):
        save_image_batch(i, np.stack(d, axis=0))

def filtering_goal(data_frame):
    return data_frame[(data_frame['tstep_reverse']==0)]

if __name__ == '__main__':
    hp = AttrDict(
        image_size_beforecrop=[56, 72],
        data_dir='/home/yanlaiyang/azure_spt/spt_data/trainingdata/robonetv2/vr_record_applied_actions_robonetv2/bww',
        n_worker=0,
        stack_goal_images=4,
    )

    dataset = LMDB_Dataset_Goal(hp, phase='train')
    loader = dataset.get_data_loader(1)

    import time
    import hashlib

    measure_time(loader)
    save_goal_images(loader)
    count_hashes(loader)

