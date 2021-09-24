import os
import shutil
import pickle as pkl
import cv2
import copy
from widowx_envs.utils.utils import AttrDict

class RawSaver():
    def __init__(self, save_dir, ngroup=1000):
        self.save_dir = save_dir
        self.ngroup = ngroup

    def save_traj(self, itr, agent_data=None, obs_dict=None, policy_outputs=None):
        igrp = itr // self.ngroup
        group_folder = os.path.join(self.save_dir , 'raw/traj_group{}'.format(igrp))
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        traj_folder = os.path.join(group_folder , 'traj{}'.format(itr))
        if os.path.exists(traj_folder):
            print('trajectory folder {} already exists, deleting the folder'.format(traj_folder))
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        print('creating: ', traj_folder)

        if 'images' in obs_dict:
            images = obs_dict['images']
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            save_dicts = []
            for t in range(T):
                for i in range(n_cams):
                    args = AttrDict(i=i, images=images, t=t, traj_folder=traj_folder)
                    save_single(args)
                    # save_dicts.append(args)
            # p = Pool(3)
            # print('saved {} total images'.format(sum(tqdm(p.imap_unordered(save_single, save_dicts), total=len(save_dicts)))))
        if 'depth_images' in obs_dict:
            depth_images = obs_dict['depth_images']
            T, n_cams = depth_images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/depth_images{}'.format(i))
            save_dicts = []
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/depth_images{}/im_{}.png'.format(traj_folder, i, t), depth_images[t, i])

        if agent_data is not None:
            with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(agent_data, file)
        if obs_dict is not None:
            obs_dict_no_image = copy.deepcopy(obs_dict)
            obs_dict_no_image.pop('images')
            with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(obs_dict_no_image, file)
        if policy_outputs is not None:
            with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(policy_outputs, file)

def save_single(arg):
    cv2.imwrite('{}/images{}/im_{}.jpg'.format(arg.traj_folder, arg.i, arg.t), arg.images[arg.t, arg.i, :, :, ::-1])
    return True


