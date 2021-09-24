
import glob
import os
import numpy as np
import copy
import cv2
import argparse
import shutil
from widowx_envs.utils.utils import np_unstack
# import moviepy as mpy
import moviepy.editor as mpy


def save_worker(traj_path, args):
    env_obs = pkl.load(open('{}/obs_dict.pkl'.format(traj_path), 'rb'), encoding='latin1')

    n_cams = len(glob.glob('{}/images*'.format(traj_path)))
    file_ending = glob.glob('{}/images0/*'.format(traj_path))[0].split('.')[-1]
    T = min([len(glob.glob('{}/images{}/*'.format(traj_path, i))) for i in range(n_cams)])

    state_time_indices = range(T)

    if len(args.save_resolution) == 0:
        height, width = cv2.imread('{}/images0/im_0.{}'.format(traj_path, file_ending)).shape[:2]
    else:
        height, width = args.save_resolution
    env_obs['images'] = np.zeros((T, n_cams, height, width, 3), dtype=np.uint8)

    for icam in range(n_cams):
        for time in state_time_indices:
            if len(args.save_resolution) != 0:
                img = cv2.imread('{}/images{}/im_{}.{}'.format(traj_path, icam, time, file_ending))
                env_obs['images'][time, icam] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            else:
                env_obs['images'][time, icam] = cv2.imread('{}/images{}/im_{}.{}'.format(traj_path, icam, time, file_ending))

    return env_obs['images']

def load_groups(trajs, traj_groups):
    global t, x
    for group in traj_groups:
        group_trajs = glob.glob('{}/*'.format(group))
        trajs.extend(group_trajs)

def run(args):
    trajs, annotations_loaded = [], 0

    datetime_groups = glob.glob(args.input_folder + "/*")
    print('found {} datetime groups!'.format(len(datetime_groups)))
    print(datetime_groups)
    for dg in datetime_groups:
        print('processing datatime group', dg)
        traj_groups = glob.glob(dg + "/raw/traj_group*")
        print('found {} traj groups!'.format(len(traj_groups)))
        load_groups(trajs, traj_groups)

    print('Loaded {} trajectories with {} annotations!'.format(len(trajs), annotations_loaded))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    # elif input_fn('path {} exists, should folder be deleted? (y/n): '.format(args.output_folder)) == 'y':
    else:
        shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)
    print('saving outputs to', args.output_folder)

    image_list = []
    for i, traj in enumerate(trajs):
        image_list.append(save_worker(traj, args))

        if len(image_list) == 3:
            make_gifs(args, i, image_list)
            image_list = []
        if i == 20:
            break


def make_gifs(args, i, image_list):
    padded = []
    maxT = max([im.shape[0] for im in image_list])

    for im in image_list:
        zerosshape = [maxT - im.shape[0]] + list(image_list[0].shape[1:])
        zeros = np.zeros(zerosshape)
        padded.append(np.concatenate([im, zeros], axis=0))

    image_rows = np.stack(padded, axis=0)
    image_rows = np.concatenate(np_unstack(image_rows, axis=2), axis=3)
    image_rows = np.concatenate(np_unstack(image_rows, axis=0), axis=2)

    # for n in range(image_list[0].shape[1]):
    npy_to_gif(np_unstack(image_rows, 0), args.output_folder + '/batch{}.gif'.format(i))


def npy_to_gif(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])
    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

if __name__ == '__main__':
    import pickle as pkl
    input_fn = input

    parser = argparse.ArgumentParser(description="converts dataset from pkl format to lmdb")
    parser.add_argument('input_folder', type=str, help='where raw files are stored')
    parser.add_argument('--output_folder', type=str, help='where to save', default='')
    parser.add_argument('--save_resolution', nargs='+', default=[], help='save resolution to use, otherwise take image orignal resolution', type=int)

    args = parser.parse_args()

    if args.output_folder == '':
        args.output_folder = os.path.join(args.input_folder, 'gifs')
    else:
        args.output_folder = os.path.join(args.output_folder, 'gifs')

    args.input_folder, args.output_folder = [os.path.expanduser(x) for x in (args.input_folder, args.output_folder)]

    run(args)