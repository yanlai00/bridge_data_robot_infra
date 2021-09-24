import torch.utils.data as data
from widowx_envs.utils.utils import Configurable, AttrDict


class BaseVideoDataset(data.Dataset, Configurable):
    def __init__(self, data_conf, phase='train', shuffle=True):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """

        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        self.phase = phase
        self.data_conf = data_conf
        self.shuffle = shuffle and phase == 'train'

        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    def _default_hparams(self):
        default_dict = AttrDict(
            n_worker=10,
        )
        return AttrDict(default_dict)

    def get_data_loader(self, batch_size):
        print('datadir {}, len {} dataset {}'.format(self.data_conf.data_dir, self.phase, len(self)))
        print('data loader nworkers', self._hp.n_worker)
        return data.DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self._hp.n_worker, drop_last=True)
