from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
from widowx_envs.policies.scripted_grasp import GraspPolicy
from rlkit.envs.wrappers import NormalizedBoxEnv

import numpy as np
import os
import argparse
import datetime
from PIL import Image

import logging
import pickle
logging.getLogger('robot_logger').setLevel(logging.WARN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-save-directory", type=str, default="/root/code/grasping_data")
    parser.add_argument("--num_trajectories", type=int, default=50000)
    parser.add_argument("--noise_std", type=float, default=0.0)
    args = parser.parse_args()

    time_string = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    directory_name = time_string + "_noise_{}".format(args.noise_std)
    save_dir = os.path.join(args.data_save_directory, directory_name)
    os.makedirs(save_dir)

    env = NormalizedBoxEnv(GraspWidowXEnv(
        {'workspace_rotation_angle_z': -1.57,
         'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': True}
    ))

    # env.reset()
    scripted_policy = GraspPolicy(env)
    pick_point_low = env._low_bound[:3]
    pick_point_high = env._high_bound[:3]
    drop_point = np.asarray([0., -0.25, 0.032])
    scripted_policy.reset(pick_point=drop_point)
    o = env.reset()

    for i in range(args.num_trajectories):
        print('traj #{}'.format(i))
        scripted_policy.drop_object(drop_point)
        obs = env.reset()
        pick_point = drop_point
        drop_point = np.random.uniform(low=pick_point_low, high=pick_point_high)
        drop_point[2] = 0.030
        # pick_point = np.concatenate((pick_point, [0.03]))
        # pick_point = np.asarray([0., -0.25, 0.032])
        scripted_policy.reset(pick_point=pick_point)

        # print(o.keys())
        time_string = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        current_save_dir = os.path.join(save_dir, time_string)
        os.makedirs(current_save_dir)

        filename = 'episode_{}.pkl'.format(time_string)
        filepath = os.path.join(current_save_dir, filename)
        transitions = []
        full_image = obs['full_image']
        obs.pop('full_image')

        for j in range(15):
            # print('is gripper open', env.is_gripper_open)
            # print('tstep #{}'.format(j))

            action, agent_info = scripted_policy.get_action()
            action = np.random.normal(loc=action, scale=args.noise_std)
            action = np.clip(action, -1.0, 1.0)
            next_obs, rew, done, info = env.step(action)

            full_image_next = next_obs['full_image']
            next_obs.pop('full_image')

            # import IPython; IPython.embed()
            im = Image.fromarray(full_image)
            imfilepath = os.path.join(current_save_dir, '{}.jpeg'.format(j+1))
            im.save(imfilepath)

            transition = {'observation': obs,
                          'next_observation': next_obs,
                          'action': action,
                          'reward': rew,
                          'done': done,
                          'info': info
                          }

            transitions.append(transition)
            obs = next_obs
            full_image = full_image_next

        with open(filepath, 'wb') as handle:
            pickle.dump(transitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print('ee', o['ee_coord'])
            # print('joints', o['joints'])
            # print('state', o['state'])
