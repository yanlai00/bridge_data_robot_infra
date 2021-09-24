import sys
from widowx_envs.run_data_collection import DataCollectionManager
import os

class SPTDataCollectionManager(DataCollectionManager):
    def set_paths(self, hyperparams):
        """
        set two directories:
            log_dir is for experiment logs, visuals, tensorboards stuff etc.
            data_save_dir is for collected datasets
            the subpath after the experiments folder is appended to the $VMPC_DATA and $VMPC_EXP directories respectively
        """
        assert 'experiments' in self.args.experiment
        subpath = hyperparams['current_dir'].partition('experiments')[2]
        hyperparams['data_save_dir'] = os.path.join(os.environ['DATA'] + '/spt_trainingdata',  subpath.strip("/"), self.save_dir_prefix.strip("/"))
        if self.time_prefix != "":
            hyperparams['data_save_dir'] = hyperparams['data_save_dir'] + '/' + self.time_prefix
        hyperparams['log_dir'] = os.path.join(os.environ['EXP'] + '/spt_experiments', subpath.strip("/"),  self.save_dir_prefix.strip("/"))
        print('setting data_save_dir to', hyperparams['data_save_dir'])
        print('setting log_dir to', hyperparams['log_dir'])
        self.hyperparams = hyperparams

if __name__ == '__main__':
    c = SPTDataCollectionManager()
    c.run()






