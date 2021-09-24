import os
import matplotlib
matplotlib.use('agg')
from widowx_envs.utils.datautils.raw2robonet import save_hdf5
import json

class RoboNetSaver(object):
    def __init__(self, save_dir, metadata_file):
        self.save_dir = os.path.join(save_dir, 'robonet')
        os.makedirs(self.save_dir)
        self.meta_data_dict = json.load(open(metadata_file, 'r'))

    def save_traj(self, itr, agent_data, obs, policy_out):
        file_name = os.path.join(self.save_dir, 'traj{}.hdf5'.format(itr))
        print('saving ', file_name)
        save_hdf5({}, file_name, obs, policy_out, agent_data, self.meta_data_dict)




