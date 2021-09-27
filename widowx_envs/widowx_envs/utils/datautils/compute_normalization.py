import numpy as np
from widowx_envs.utils.datautils.lmdb_dataloader import LMDB_Dataset
from widowx_envs.utils.utils import AttrDict
import os

import pickle as pkl
import argparse

def compute_dataset_normalization(data_dir, no_last_dim_norm=True, separate_tsteps=False):
    hp = AttrDict(data_dir=data_dir)
    batch_size = 10

    hp.update(AttrDict(
        image_size_beforecrop = [56, 72],
        random_crop=[48, 64],
        sel_camera=0
    ))
    loader = LMDB_Dataset(hp, phase='train').get_data_loader(batch_size)

    states_list = []
    action_list = []

    for i_batch, sample_batched in enumerate(loader):
        states_list.append(sample_batched['states'])
        action_list.append(sample_batched['actions'])
        if i_batch == 1500:
            break

    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(action_list, axis=0)


    print('state dim: ', states.shape)
    print('action dim: ', actions.shape)
    if actions.shape[0] < 1000:
        print('Very few examples found!!!')
        # import pdb; pdb.set_trace()

    dict = {
    'states_mean' : np.mean(states, axis=0),
    'states_std' : np.std(states, axis=0),
    'actions_mean': np.mean(actions, axis=0),
    'actions_std': np.std(actions, axis=0),
    }

    for dim in range(states.shape[1]):
        if dict['states_mean'][dim] == 0 and dict['states_std'][dim] == 0:
            dict['states_mean'][dim] = 0
            dict['states_std'][dim] = 1
            print('##################################')
            print('not normalizing state dim {}, since mean and std are zero!!'.format(dim))
            print('##################################')

    for dim in range(actions.shape[1]):
        if dict['actions_mean'][dim] == 0 and dict['actions_std'][dim] == 0:
            dict['actions_mean'][dim] = 0
            dict['actions_std'][dim] = 1
            print('##################################')
            print('not normalizing action dim {}, since mean and std are zero!!'.format(dim))
            print('##################################')

    if no_last_dim_norm:
        print('##################################')
        print('not normalizing grasp action!')
        print('##################################')
        dict['actions_mean'][-1] = 0
        dict['actions_std'][-1] = 1

    print(dict)

    print('saved to ', os.path.join(data_dir, 'normalizing_params.pkl'))
    pkl.dump(dict, open(os.path.join(data_dir, 'normalizing_params.pkl'), 'wb'), protocol=2)


def compute_discretization_pivots_means(actions, dict):

    normalized_actions = (actions - dict['actions_mean'])/dict['actions_std']

    pivots = []
    means = []
    n_discrete = 10
    for i in range(normalized_actions.shape[1]):
        action_dim = normalized_actions[:, i]
        sorted_actions = np.sort(action_dim)

        N = sorted_actions.shape[0]
        means_per_dim = []
        pivots_per_dim = []
        chunk_size = N//n_discrete
        for n in range(n_discrete):
            action_chunk = sorted_actions[n * chunk_size: (n + 1) * chunk_size]
            means_per_dim.append(np.mean(action_chunk))
            if n < n_discrete - 1:
                pivots_per_dim.append(action_chunk[-1])

        means_per_dim = np.array(means_per_dim)
        means.append(means_per_dim)
        pivots_per_dim = np.array(pivots_per_dim)
        pivots.append(pivots_per_dim)

    means = np.stack(means, 0)
    pivots = np.stack(pivots, 0)
    return means, pivots

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to dataset')
    parser.add_argument('--no_last_dim_norm', type=int, help='whether to normalize the last dim, typically used for the action (if not autograsp)', default=1)
    parser.add_argument('--discretization', action='store_true', help='compute discretization means and pivots')
    parser.add_argument('--robonet', type=int, help='use robotnet dataloader', default=1)
    args = parser.parse_args()
    compute_dataset_normalization(args.data_dir, bool(args.no_last_dim_norm), args.discretization)

