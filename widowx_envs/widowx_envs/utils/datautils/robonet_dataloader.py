from widowx_envs.utils.datautils.robonet_dataset import RoboNetDataset
import numpy as np
from widowx_envs.utils.datautils.base_data_loader import BaseVideoDataset
from widowx_envs.utils.datautils.metadata_helper import load_metadata
from widowx_envs.utils.datautils.hdf5_loader import HDF5Loader
from widowx_envs.utils.datautils.data_augmentation import get_random_crop, get_random_color_aug
from widowx_envs.utils.utils import AttrDict
import os
import random
import torch


class FilteredRoboNetDataset(RoboNetDataset, BaseVideoDataset):
    def __init__(self, data_conf, phase='train', shuffle=True):
        BaseVideoDataset.__init__(self, data_conf, phase, shuffle=shuffle)
        self.phase = phase
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        if isinstance(self._hp.data_dir, list):
            data_dirs = self._hp.data_dir
        else:
            data_dirs = [self._hp.data_dir]
        metadata_list = []
        for data_dir in data_dirs:
            if self._hp.splits is None: # if there are separate folder for train val test
                metadata = load_metadata(os.path.join(data_dir, phase))
            else:
                metadata = load_metadata(data_dir)
            if self._hp.robot_list is not None:
                metadata = metadata[metadata['robot'].frame.isin(self._hp.robot_list)]
            metadata_list.append(metadata)
        data_conf['img_size'] = self._hp.image_size_beforecrop
        RoboNetDataset.__init__(self, metadata_list, phase, data_conf)
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

    def _default_hparams(self):
        default_dict = AttrDict(
            name='',
            n_worker=10,
            color_augmentation=False,
            random_crop=False,
            image_size_beforecrop=None,
            T=None,
            robot_list=None,
            data_dir=None,
            sel_camera=-1,  # -1 means selecting all cameras,
            target_adim=7,
            target_sdim=7,
            normalize_images=False,
            sel_len=-1,  # number of time steps for contigous sequence that is shifted within sequeence of T randomly
        )
        parent_params = AttrDict(super()._get_default_hparams())
        parent_params.update(default_dict)
        return parent_params

    def _check_params(self, sources):
        max_steps = max([max(m.frame['img_T']) for m in sources])
        print('maxsteps', max_steps)
        assert self._hp['T'] >= 0, "can't load less than 0 frames!"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, metadata = self._data[idx]

        loader = HDF5Loader(file_name, metadata, self._hp)
        # get action/state vectors
        states, actions = loader.load_states(), loader.load_actions()

        data_dict = AttrDict()
        # get camera and slice states if T is set
        if self._hp.sel_camera == 'random':

            cam_ind = random.randint(0, metadata['ncam'])
            cam_indices = [cam_ind]
            data_dict['camera_ind'] = np.array([cam_ind])
        elif self._hp.sel_camera == -1:
            cam_indices = range(metadata['ncam'])
        elif isinstance(self._hp.sel_camera, tuple):
            cam_indices = self._hp.sel_camera
        elif isinstance(self._hp.sel_camera, int):
            cam_indices = [self._hp.sel_camera]
        else:
            raise NotImplementedError

        img_len = metadata['img_T']
        if self._hp.sel_len != -1:
            assert img_len > self._hp.sel_len
            start_time = random.randint(0, img_len - self._hp.sel_len)

            images = np.stack([loader.load_video(ind, start_time=start_time, n_load=self._hp.sel_len)
                                     for ind in cam_indices], axis=1)
            states = states[start_time:start_time + self._hp.sel_len]
            actions = actions[start_time:start_time + self._hp.sel_len - 1]
        else:
            images = np.stack([loader.load_video(ind) for ind in cam_indices], axis=1)

        images = self._proc_images(images)

        augmented_images = []
        for cam in range(images.shape[1]):
            augmented_images.append(self.apply_data_augmentation(images[:, cam]))
        images = np.stack(augmented_images, 1)
        images = images*2 - 1
        data_dict.update(AttrDict(images=images,
                             states=states.astype(np.float32),
                             actions=actions.astype(np.float32),
                             robot_type=loader.load_robot_id(self._hp.robot_list),
                             tlen=min(img_len, self._hp.T)
                             ))

        for k, v in data_dict.items():
            if k in ['robot_type', 'tlen', 'camera_ind']:
                continue
            if k == 'actions':
                desired_T = self._hp.T - 1 # actions need to be shorter by one since they need to have a start and end-state!
            else:
                desired_T = self._hp.T
            if v.shape[0] < desired_T:
                data_dict[k] = self.pad_tensor(v, desired_T)
            else:
                data_dict[k] = v[:desired_T]

        return data_dict

    def pad_tensor(self, tensor, desired_T):
        pad = np.zeros([desired_T - tensor.shape[0]] + list(tensor.shape[1:]), dtype=np.float32)
        tensor = np.concatenate([tensor, pad], axis=0)
        return tensor

    def apply_data_augmentation(self, images):
        if self._hp.random_crop:
            if self.phase == 'train':
                images = get_random_crop(images, self._hp.random_crop)
            else:
                images = get_random_crop(images, self._hp.random_crop, center_crop=True)
        if self._hp.color_augmentation and self.phase == 'train':
            images = get_random_color_aug(images, self._hp.color_augmentation)
        return images


if __name__ == '__main__':

    # hp = AttrDict(
    #     name='robonet_sawyer',
    #     T=31,
    #     robot_list=['sawyer'],
    #     train_val_split=[0.8, 0.1, 0.1],
    #     # train_val_split=[0.95, 0.025, 0.025],
    #     # color_augmentation=0.3,
    #     random_crop=True,
    #     data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet_sampler/hdf5',
    #     # data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/hdf5'
    #     random_camera = False
    # )

    hp = AttrDict(
        T=25,
        image_size_beforecrop=[112, 144],
        random_crop=[96, 128],
        splits=None,
        robot_list=['widowx'],
        data_dir=[os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_control/bww_grasp_pen/hdf5'],
        sel_camera='random'
    )

    loader = FilteredRoboNetDataset(hp, phase='train').get_data_loader(8)
    from semiparametrictransfer.data_sets.data_utils.test_datasets import measure_time, make_gifs
    # measure_time(loader)
    make_gifs(loader)

    for i_batch, sample_batched in enumerate(loader):
        images = np.asarray(sample_batched['images'])
        print(i_batch)

