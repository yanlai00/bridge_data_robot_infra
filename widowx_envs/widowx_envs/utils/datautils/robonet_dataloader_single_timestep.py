from widowx_envs.utils.datautils.base_data_loader import BaseVideoDataset
from widowx_envs.utils.datautils.robonet_dataset import RoboNetDataset
from widowx_envs.utils.datautils.metadata_helper import load_metadata
from widowx_envs.utils.datautils.hdf5_loader import HDF5Loader
from widowx_envs.utils.datautils.data_augmentation import get_random_crop, get_random_color_aug
from widowx_envs.utils.utils import AttrDict


class FilteredRoboNetDatasetSingleTimeStep(RoboNetDataset, BaseVideoDataset):
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
            if self._hp.sel_camera != 'random':
                metadata = metadata[metadata['camera_index'] == self._hp.sel_camera]
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
            sel_camera='random',  # -1 means selecting all cameras,
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

        cam_index = metadata['camera_index']
        final_tstep = metadata['img_T'] - 1

        path = '/'.join(str.split(file_name, '/')[:-1])
        final_filename = os.path.join(path, str.split(str.split(file_name, '/')[-1], '_')[0] + '_t{}_cam{}.hdf5'.format(final_tstep, cam_index))
        final_loader = HDF5Loader(final_filename, metadata, self._hp, check_hash=False)
        # get action/state vectors
        states, actions = loader.load_states(), loader.load_actions()

        data_dict = AttrDict()

        images = self.get_image(loader)
        final_image = self.get_image(final_loader)

        data_dict.update(AttrDict(images=images,
                                 states=states.astype(np.float32),
                                 actions=actions.astype(np.float32),
                                 robot_type=loader.load_robot_id(self._hp.robot_list),
                                 final_image=final_image,
                                 camera_ind=cam_index
                                 ))

        return data_dict

    def get_image(self, loader):
        image = loader.load_image()
        image = self._proc_images(image)
        image = self.apply_data_augmentation(image)
        image = image * 2 - 1
        return image

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
        color_augmentation=0.1,
        splits=None,
        robot_list=['widowx'],
        data_dir=[os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_pen/hdf5_separate_tsteps'],
        # sel_camera='random',
        # n_worker=0  ########
    )

    loader = FilteredRoboNetDatasetSingleTimeStep(hp, phase='train').get_data_loader(8)
    from semiparametrictransfer.data_sets.data_utils.test_datasets import measure_time, make_gifs
    measure_time(loader)

    # for i_batch, sample_batched in enumerate(loader):
    #     images = np.asarray(sample_batched['images'])
    #     final_images = np.asarray(sample_batched['final_frame'])
    #     print('images shape', images.shape)
    #     print('final images shape', final_images.shape)
    #     print(i_batch)
    #


