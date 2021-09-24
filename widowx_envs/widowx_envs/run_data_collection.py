import argparse
import imp
import json
from widowx_envs.utils.utils import ask_confirm, save_config
import datetime
import os
from widowx_envs.trajectory_collector import TrajectoryCollector

class DataCollectionManager(object):
    def __init__(self, save_dir_prefix='', args_in=None, hyperparams=None):
        """
        :param save_dir_prefix: will be added to the experiment data and training data save paths
        specified in the variables $VMPC_DATA and $VMPC_EXP
        :param args_in:
        :param hyperparams:
        """
        parser = argparse.ArgumentParser(description='run parallel data collection')
        parser.add_argument('experiment', type=str, help='path to the experiment configuraiton file including mod_hyper.py')
        parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
        parser.add_argument('--ngpu', type=int, help='the number of gpus to use', default=1)
        parser.add_argument('--prefix', default='', help='prefixes for data and exp dirs')

        args = parser.parse_args(args_in)
        if hyperparams is None:
            hyperparams = imp.load_source('hyperparams', args.experiment)
            self.hyperparams = hyperparams.config
        else:
            self.hyperparams = hyperparams
        self.args = args
        if save_dir_prefix is not '':
            self.save_dir_prefix = save_dir_prefix
        else:
            self.save_dir_prefix = args.prefix
        self.time_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def run(self):
        hyperparams = self.hyperparams

        self.set_paths(hyperparams)
        save_config(hyperparams, hyperparams['data_save_dir'])

        if 'collection_metadata' in hyperparams:
            metadata_destination = hyperparams['data_save_dir']
            meta_data_dict = json.load(open(hyperparams['collection_metadata'], 'r'))
            meta_data_dict['date_time'] = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            print('#################################################')
            print('#################################################')
            for k, v in meta_data_dict.items():
                print('{}: {}'.format(k,v))
            ask_confirm('Is the meta data correct? press y/n')
            os.system('cp {} {}'.format(hyperparams['collection_metadata'], metadata_destination))

        s = TrajectoryCollector(hyperparams)
        s.run()

    def set_paths(self, hyperparams):
        """
        set two directories:
            log_dir is for experiment logs, visuals, tensorboards stuff etc.
            data_save_dir is for collected datasets
            the subpath after the experiments folder is appended to the $VMPC_DATA and $VMPC_EXP directories respectively
        """
        project_key = 'robonetv2'

        assert 'experiments' in self.args.experiment
        subpath = hyperparams['current_dir'].partition('experiments')[2]
        hyperparams['data_save_dir'] = os.path.join(os.environ['DATA'], project_key,  subpath.strip("/"), self.save_dir_prefix.strip("/"))
        if self.time_prefix != "":
            hyperparams['data_save_dir'] = hyperparams['data_save_dir'] + '/' + self.time_prefix
        hyperparams['log_dir'] = os.path.join(os.environ['EXP'], project_key, subpath.strip("/"),  self.save_dir_prefix.strip("/"))
        print('setting data_save_dir to', hyperparams['data_save_dir'])
        print('setting log_dir to', hyperparams['log_dir'])
        self.hyperparams = hyperparams

if __name__ == '__main__':
    DataCollectionManager().run()
