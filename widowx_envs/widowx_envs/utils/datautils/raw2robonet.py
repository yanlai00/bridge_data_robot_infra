import h5py
import cv2
import numpy as np
import copy
import imageio
from multiprocessing import Pool, Manager
import functools
from tqdm import tqdm
from widowx_envs.utils.datautils.compute_normalization import compute_dataset_normalization
from widowx_envs.utils.sync import ManagedSyncCounter

import argparse
import glob
import json
import random
import sys
import os
import shutil
import math

MANDATORY_KEYS = ['camera_configuration', 'policy_desc', 'bin_type', 'bin_insert',
                                'robot', 'gripper', 'background', 'action_space', 'object_classes', 'primitives', 'camera_type']


def serialize_image(img):
    assert img.dtype == np.uint8, "Must be uint8!"
    return cv2.imencode('.jpg', img)[1]


def serialize_video(imgs, temp_name_append):
    mp4_name = './temp{}.mp4'.format(temp_name_append)
    try:
        assert imgs.dtype == np.uint8, "Must be uint8 array!"
        assert not os.path.exists(mp4_name), "file {} exists!".format(mp4_name)
        # this is a hack to ensure imageio succesfully saves as a mp4 (instead of getting encoding confused)
        writer = imageio.get_writer(mp4_name)
        [writer.append_data(i[:, :, ::-1]) for i in imgs]
        writer.close()

        f = open(mp4_name, 'rb')
        buf = f.read()
        f.close()
    finally:
        if os.path.exists(mp4_name):
            os.remove(mp4_name)

    return np.frombuffer(buf, dtype=np.uint8)


def save_dict(data_container, dict_group, video_encoding, t_index):
    for k, d in data_container.items():
        if 'images' == k:
            T, n_cams = d.shape[:2]
            dict_group.attrs['n_cams'] = n_cams
            for n in range(n_cams):
                dict_group.attrs['cam_encoding'] = video_encoding
                cam_group = dict_group.create_group("cam{}_video".format(n))
                if video_encoding == 'mp4':
                    data = cam_group.create_dataset("frames", data=serialize_video(d[:, n], t_index))
                    data.attrs['shape'] = d[0, n].shape
                    data.attrs['T'] = d.shape[0]
                    data.attrs['image_format'] = 'RGB'
                elif video_encoding == 'jpeg':
                    for t in range(T):
                        data = cam_group.create_dataset("frame{}".format(t), data=serialize_image(d[t, n]))
                        data.attrs['shape'] = d[t, n].shape
                        data.attrs['image_format'] = 'RGB'
                else:
                    raise ValueError
        elif 'image' in k:
            data = dict_group.create_dataset(k, data=serialize_image(d))
            data.attrs['shape'] = d.shape
        elif d is None:
            continue
        else:
            dict_group.create_dataset(k, data=d)


def save_hdf5(args, filename, env_obs, policy_out, agent_data, meta_data, video_encoding='mp4', t_index=None):
    if t_index is None:
        t_index = random.randint(0, 9999999)
    # meta-data includes calibration "number", policy "type" descriptor, environment bounds
    with h5py.File(filename, 'w') as f:
        f.create_dataset('file_version', data='0.1.0')
        [save_dict(data_container, f.create_group(name), video_encoding, t_index) for data_container, name in zip([env_obs, agent_data], ['env', 'misc'])]

        policy_dict = {}
        if not args.separate_tsteps:
            first_keys = list(policy_out[0].keys())
            for k in first_keys:
                assert all([k in p for p in policy_out[1:]]), "hdf5 format requires keys must be uniform across time!"
                policy_dict[k] = np.concatenate([p[k][None] for p in policy_out], axis=0)
        else:
            policy_dict = policy_out

        if 'convert_rel_to_abs_action' in args:
            if args.convert_rel_to_abs_action:
                policy_dict['actions'] = convert_rel_to_abs_action(policy_dict['actions'], env_obs)
        save_dict(policy_dict, f.create_group('policy'), video_encoding, t_index)

        meta_data_group = f.create_group('metadata')
        for mandatory_key in MANDATORY_KEYS:
            assert mandatory_key in meta_data
        for k in meta_data.keys():
            meta_data_group.attrs[k] = meta_data[k]

def convert_rel_to_abs_action(actions, env_obs):
    """
    assume gripper actions are the last dimension.
    """
    # assume that the last dimension of the state is the gripper position
    state = env_obs['state']
    if len(state.shape) == 2: # if not using separate_tsteps
        for t in range(state.shape[0]):
            if state[t, -1] not in [-1, 1, 0]:
                raise ValueError('gripper state must be either -1, 1 or 0!')
        actions[:, -1] = state[:-1, -1]
    else:  # if using separate_tsteps
        import pdb; pdb.set_trace()
        actions[-1] = state[-1]
    return actions

def save_worker(traj_data, args, cntr, group_name=''):
    t_index = random.randint(0, 9999999)
    traj_path, meta_data = traj_data

    try:
        env_obs = pkl.load(open('{}/obs_dict.pkl'.format(traj_path), 'rb'), encoding='latin1')
        if meta_data['contains_annotation']:
            env_obs['bbox_annotations'] = pkl.load(open('{}/annotation_array.pkl'.format(traj_path), 'rb'), encoding='latin1')

        n_cams = len(glob.glob('{}/images*'.format(traj_path)))
        if 'camera_variation' not in meta_data:
            meta_data['camera_variation'] = 0.0
        file_ending = glob.glob('{}/images0/*'.format(traj_path))[0].split('.')[-1]
        T = min([len(glob.glob('{}/images{}/*'.format(traj_path, i))) for i in range(n_cams)])

        if args.task_stage != -1:  # only save time-steps from desired task stage
            task_stage = np.array(env_obs['task_stage'])
            state_time_indices = np.where(task_stage == args.task_stage)[0]
            # check that time steps are consecutiv
            for i, j in zip(state_time_indices[1:], state_time_indices):
                assert i - j == 1, 'stage time steps are not consecutive!'
            print('stage length', len(state_time_indices))
        else:
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

        policy_out = pkl.load(open('{}/policy_out.pkl'.format(traj_path), 'rb'), encoding='latin1')
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(traj_path), 'rb'), encoding='latin1')

        if args.task_stage != -1:  # only take time-steps from desired task stage
            for key, value in env_obs.items():
                if isinstance(value, np.ndarray):
                    if value.shape[0] == T:
                        env_obs[key] = value[state_time_indices]
            policy_out = [policy_out[t] for t in state_time_indices[:-1]]

        def store_in_metadata_if_exists(key):  
            if key in agent_data:
                meta_data[key] = agent_data.pop(key)
        [store_in_metadata_if_exists(k) for k in ['goal_reached', 'term_t']]

        c = cntr.ret_increment
        save_folder = args.output_folder

        if not args.separate_tsteps:
            filename = '{}/{}traj{}.hdf5'.format(save_folder, group_name, c)
            save_hdf5(args, filename, env_obs, policy_out, agent_data, meta_data, video_encoding, t_index)
        else:
            state_time_indices = state_time_indices[:len(policy_out)]
            for tstep in state_time_indices:
                env_obs_single = {}
                for key, value in env_obs.items():
                    if key == 'reset_state':  # don't save the reset state since it doesn't have temporal dimension
                        continue
                    if key == 'images':
                        env_obs_single['image'] = value[tstep]
                    else:
                        env_obs_single[key] = value[tstep]
                policy_out_single = policy_out[tstep]
                for icam in range(n_cams):
                    filename = '{}/{}traj{}_t{}_cam{}.hdf5'.format(save_folder, group_name, c, tstep, icam)
                    per_cam_env_obs_single = copy.deepcopy(env_obs_single)
                    per_cam_env_obs_single['image'] = env_obs_single['image'][icam]
                    meta_data['camera_index'] = icam
                    meta_data['tstep'] = tstep
                    meta_data['img_T'] = len(state_time_indices)
                    save_hdf5(args, filename, per_cam_env_obs_single, policy_out_single, agent_data, meta_data, video_encoding, t_index)
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


def load_groups(trajs, traj_groups, annotations_loaded, meta_data_dict):
    global t, x
    for group in traj_groups:
        group_trajs = glob.glob('{}/*'.format(group))
        for t in group_trajs:
            traj_meta_data = copy.deepcopy(meta_data_dict)
            traj_meta_data['object_batch'] = group
            if os.path.exists('{}/annotation_array.pkl'.format(t)):
                traj_meta_data['contains_annotation'] = True
                annotations_loaded += 1
            else:
                traj_meta_data['contains_annotation'] = False

            if isinstance(traj_meta_data['object_classes'], str):
                traj_meta_data['object_classes'] = traj_meta_data['object_classes'].split("+")

            assert all([k in traj_meta_data for k in MANDATORY_KEYS]), 'metadata for {} is missing keys!'.format(t)
            assert isinstance(traj_meta_data['object_classes'], list), "did not split object classes!"
            assert all(
                [isinstance(x, str) for x in traj_meta_data['object_classes']]), 'object classes is not a string!'
            trajs.append((t, traj_meta_data))
    return annotations_loaded


def split_files_trainval(output_folder, separate_tsteps):
    globbed_files = glob.glob(output_folder + '/*.hdf5')

    # make sure that we're moving all time-steps and cameras from a particular directory
    files = set()
    for file in globbed_files:
        filename_ending = str.split(file, '/')[-1]
        if '_' in filename_ending:
            filename_ending = str.split(filename_ending, '_')[0]
        files.add(filename_ending)

    files = list(files)
    print("number of trajecotries", len(files))

    random.shuffle(files)
    nfiles = len(files)
    phases = ['train', 'val', 'test']
    n_files_cum = np.cumsum((np.array(args.train_val_split) * nfiles).astype(np.int))
    train_files = files[0 : n_files_cum[0]]
    val_files = files[n_files_cum[0] : n_files_cum[1]]
    test_files = files[n_files_cum[2]:]
    assert len(train_files) + len(val_files) + len(test_files) == len(files)

    for phase, phase_files in zip(phases, [train_files, val_files, test_files]):
        print(phase, len(phase_files))
        dest_folder = os.path.join(args.output_folder, phase)
        os.makedirs(dest_folder)
        for file in phase_files:
            if separate_tsteps:
                selected_files = glob.glob(output_folder + '/{}_*'.format(file))
                print(output_folder + '/{}_*'.format(file))
            else:
                selected_files = glob.glob(output_folder + '/' + file)
            if len(selected_files) == 0:
                print('no files found')
                import pdb; pdb.set_trace()
            for selfile in selected_files:
                shutil.move(selfile, dest_folder)
            print("moved {} files".format(len(selected_files)))


def run(args):
    global video_encoding, i
    if args.video_jpeg_encoding:
        video_encoding = 'jpeg'
    else:
        video_encoding = 'mp4'
        if len(glob.glob('./temp*.mp4')) != 0:
            print("Please delete all temp*.mp4 files! (needed for saving)")
            raise EnvironmentError
    trajs, annotations_loaded = [], 0
    traj_groups = glob.glob(args.input_folder + "/traj_group*")
    if os.path.isfile('{}/collection_metadata.json'.format(args.input_folder)):
        general_meta_data_dict = json.load(open('{}/collection_metadata.json'.format(args.input_folder), 'r'))
        print('found general metadatadict.')
    else:
        general_meta_data_dict = None
    if traj_groups == []:  # in case we're dealing with a date-time folders
        datetime_groups = glob.glob(args.input_folder + "/*")
        print('found {} datetime groups!'.format(len(datetime_groups)))
        print(datetime_groups)
        for dg in datetime_groups:
            print('processing datatime group', dg)
            traj_groups = glob.glob(dg + "/raw/traj_group*")
            print('found {} traj groups!'.format(len(traj_groups)))
            print(traj_groups)

            metadata_dict_datetimegroup = '{}/collection_metadata.json'.format(dg)
            if os.path.exists(metadata_dict_datetimegroup):  # try to load metadatadict insdie group if it's not provided
                general_meta_data_dict = json.load(open(metadata_dict_datetimegroup, 'r'))
                print(general_meta_data_dict)
            annotations_loaded = load_groups(trajs, traj_groups, annotations_loaded, general_meta_data_dict)
    else:
        print('found {} traj groups!'.format(len(traj_groups)))
        annotations_loaded = load_groups(trajs, traj_groups, annotations_loaded, general_meta_data_dict)
    random.shuffle(trajs)
    print('Loaded {} trajectories with {} annotations!'.format(len(trajs), annotations_loaded))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    elif input_fn('path {} exists, should folder be deleted? (y/n): '.format(args.output_folder)) == 'y':
        shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)
    else:
        import sys
        sys.exit()
    print('saving outputs to', args.output_folder)
    # print('############################o')
    # print('############################o')
    # print('############################o')
    # print('############################o')
    # trajs = trajs[:20]
    # print('only using {} trajs!!!!'.format(len(trajs)))
    # import pdb; pdb.set_trace()
    cntr = ManagedSyncCounter(Manager(), args.counter)
    if args.n_workers == 1:
        saved = 0
        for i, traj in enumerate(tqdm(trajs)):
            saved += save_worker(traj, args, cntr, args.output_group_name)
        print('saved {} total trajs'.format(saved))
    else:
        map_fn = functools.partial(save_worker, args=args, cntr=cntr, group_name=args.output_group_name)
        p = Pool(args.n_workers)
        print('parallel saved {} total trajs'.format(sum(tqdm(p.imap_unordered(map_fn, trajs), total=len(trajs)))))
    if args.train_val_split:
        split_files_trainval(args.output_folder, args.separate_tsteps)
    compute_dataset_normalization(args.output_folder, separate_tsteps=args.separate_tsteps)


if __name__ == '__main__':
    if sys.version_info[0] == 2:
        import cPickle as pkl
        input_fn = raw_input
    else:
        import pickle as pkl
        input_fn = input

    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='where raw files are stored')
    parser.add_argument('--output_folder', type=str, help='where to save', default='')
    parser.add_argument('--output_group_name', type=str, default='', help='name to prepend in front of trajs')
    parser.add_argument('--video_jpeg_encoding', action='store_true', default=False, help='uses jpeg encoding for video frames instead of mp4')
    parser.add_argument('--counter', type=int, help='where to start counter', default=0)
    parser.add_argument('--n_workers', type=int, help='number of multi-threaded workers', default=1)
    parser.add_argument('--convert_rel_to_abs_action', action='store_true', help='convert grasp action (assumed to be last dimension) from relative to absolute')
    parser.add_argument('--task_stage', type=int, default=-1, help='only save time-steps from the desired task stage')
    # parser.add_argument('--save_resolution', nargs='+', default=[112, 144], help='save resolution to use, otherwise take image orignal resolution', type=int)
    parser.add_argument('--save_resolution', nargs='+', default=[], help='save resolution to use, otherwise take image orignal resolution', type=int)
    parser.add_argument('--train_val_split', nargs='+', default=[0.9, 0.1, 0], help='save trajs into seperate folders for train and val', type=float)
    parser.add_argument('--separate_tsteps', action='store_true', help='save time steps separately')

    args = parser.parse_args()

    assert args.n_workers >= 1, "can't have less than 1 worker thread!"
    if args.output_folder == '':
        args.output_folder = os.path.join(args.input_folder, 'hdf5')
    else:
        args.output_folder = os.path.join(args.output_folder, 'hdf5')
    if args.separate_tsteps:
        args.video_jpeg_encoding = 'jpeg'
        args.output_folder += '_separate_tsteps'

    args.input_folder, args.output_folder = [os.path.expanduser(x) for x in (args.input_folder, args.output_folder)]

    run(args)
    # split_files_trainval(args.output_folder, args.separate_tsteps)
