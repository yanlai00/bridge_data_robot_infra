import matplotlib.pyplot as plt

MANDATORY_KEYS = ['camera_configuration', 'policy_desc', 'robot', 'gripper', 'background', 'action_space', 'object_classes', 'primitives', 'camera_type']

EXCLUDED_KEYS = ['camera_info']

import hashlib
import lmdb
import argparse
import shutil
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import copy
import json
import imageio
import glob
import random
import os
import pickle as pkl
from widowx_envs.utils.datautils.compute_normalization import compute_dataset_normalization
import pyarrow as pa

def serialize_image(img):
    assert img.dtype == np.uint8, "Must be uint8!"
    return cv2.imencode('.jpg', img)[1]

def save_dict(data_container, dict_group, video_encoding, t_index):
    for k, d in data_container.items():
        if 'images' == k:
            T, n_cams = d.shape[:2]
            dict_group.attrs['n_cams'] = n_cams
            for n in range(n_cams):
                dict_group.attrs['cam_encoding'] = video_encoding
                cam_group = dict_group.create_group("cam{}_video".format(n))
                if video_encoding == 'jpeg':
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


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def make_domain_hash(meta_data):
    meta_data_cpy = copy.deepcopy(meta_data)
    included_keys = ['camera_index', 'camera_variation', 'environment', 'lighting', 'background']
    dict_considered_for_hash = {}
    for key in included_keys:
        if key in meta_data_cpy:
            dict_considered_for_hash[key] = meta_data_cpy[key]
        else:
            dict_considered_for_hash[key] = None
    return hashlib.sha256(dumps_pyarrow(dict_considered_for_hash)).hexdigest()

def save_worker(i, index_dict, traj_data, args, database_index, txn, db, metadata_list):
    traj_path, meta_data = copy.deepcopy(traj_data)
    meta_data['orig_policy_desc'] = meta_data['policy_desc']

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

        if len(args.down_sample) == 0:
            height, width = cv2.imread('{}/images0/im_0.{}'.format(traj_path, file_ending)).shape[:2]
        else:
            height, width = args.down_sample
        env_obs['images'] = np.zeros((T, n_cams, height, width, 3), dtype=np.uint8)

        for icam in range(n_cams):
            for time in state_time_indices:
                if len(args.down_sample) != 0:
                    img = cv2.imread('{}/images{}/im_{}.{}'.format(traj_path, icam, time, file_ending))
                    env_obs['images'][time, icam] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    env_obs['images'][time, icam] = cv2.imread('{}/images{}/im_{}.{}'.format(traj_path, icam, time, file_ending))

        policy_out = pkl.load(open('{}/policy_out.pkl'.format(traj_path), 'rb'), encoding='latin1')
        agent_data = pkl.load(open('{}/agent_data.pkl'.format(traj_path), 'rb'), encoding='latin1')

        for key in EXCLUDED_KEYS:
            if key in agent_data:
                agent_data.pop(key)

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

        state_time_indices = state_time_indices[:len(policy_out)]

        cropping_center = None
        crop_height = 56
        cropwidth = 72
        if args.select_image_croppping_pos:
            cropping_data_file = traj_path + '/cropping_center.pkl'
            if os.path.isfile(cropping_data_file):
                print('loading from ', cropping_data_file)
                cropping_pos_ratio = pkl.load(open(cropping_data_file, 'rb'), encoding='latin1')['cropping_ratio']
                # cropping_pos_ratio = cropping_center / np.array([96.0, 128.0])
            else:
                from widowx_envs.utils.datautils.annotate_object_pos import Getdesig
                coords = Getdesig(env_obs['images'][0, 0][:, :, ::-1])
                cropping_pos_ratio = np.array(coords.desig, dtype=np.float32) /  env_obs['images'][0, 0].shape[:2]
                pkl.dump({'cropping_ratio': cropping_pos_ratio}, open(cropping_data_file, 'wb'))
                print(cropping_pos_ratio)
                print('saving ', cropping_data_file)

            cropping_center = (env_obs['images'][0, 0].shape[:2] * cropping_pos_ratio).astype(np.int32)
            # cropped_image = crop_image(crop_height, cropwidth, cropping_center,
            #                            env_obs['images'][0, 0][:, :, ::-1])
            # plt.imshow(cropped_image)
            # plt.show()
        elif args.crop_at_coords:
            cropping_center = np.array(args.crop_at_coords)

        index_dict['traj{}'.format(i)] = {}

        date = meta_data['camera_variation'].split('_')[0]
        month, day = int(date.split('-')[0]), int(date.split('-')[1])
        if month > 7 or (month == 7 and day >= 23):
            latency_shift = 0
        else:
            latency_shift = 1
        
        meta_data['policy_desc'] = []
        meta_data['policy_type'] = []

        for tstep in state_time_indices:
            if tstep + latency_shift >= len(policy_out): 
                continue
            index_dict['traj{}'.format(i)]['t{}'.format(tstep)] = {}
            env_obs_single = {}
            for key, value in env_obs.items():
                if key == 'reset_state':  # don't save the reset state since it doesn't have temporal dimension
                    continue
                if key == 'images': 
                    env_obs_single['image'] = value[tstep + latency_shift]
                else:
                    env_obs_single[key] = value[tstep]

            policy_out_single = policy_out[tstep]
            for icam in range(n_cams):
                index_dict['traj{}'.format(i)]['t{}'.format(tstep)]['cam{}'.format(icam)] = database_index

                per_cam_env_obs_single = copy.deepcopy(env_obs_single)
                if cropping_center is not None:
                    image = crop_image(crop_height, cropwidth, cropping_center, env_obs_single['image'][icam])
                else:
                    image = env_obs_single['image'][icam]
                per_cam_env_obs_single['image'] = serialize_image(image)
                meta_data['camera_index'] = icam
                meta_data['tstep'] = tstep
                meta_data['img_T'] = len(state_time_indices)
                meta_data['database_index'] = database_index

                meta_data['domain_hash'] = make_domain_hash(meta_data)

                if 'policy_desc' in policy_out_single.keys():
                    meta_data['policy_desc'] = policy_out_single['policy_desc']
                if 'policy_type' in policy_out_single.keys():
                    meta_data['policy_type'] = policy_out_single['policy_type']

                for mandatory_key in MANDATORY_KEYS:
                    assert mandatory_key in meta_data

                data_dict = {
                    'env_obs': per_cam_env_obs_single,
                    'policy_out': policy_out_single,
                    'agent_data': agent_data,
                    'meta_data': copy.deepcopy(meta_data)
                }
                
                data_dict['policy_out'] = dict(data_dict['policy_out'])
                txn.put(u'{}'.format(database_index).encode('ascii'), dumps_pyarrow(data_dict))
            
                if database_index % 5000 == 0:
                    print("committing to database at {}".format(database_index))
                    txn.commit()
                    txn = db.begin(write=True)
                database_index += 1

                metadata_list.append(copy.deepcopy(meta_data))
        return index_dict, database_index, txn, db, metadata_list
    except (FileNotFoundError, NotADirectoryError):
        return False


def crop_image(target_height, target_width, cropping_center, image):
    if len(image.shape) == 3:
        imsize = image.shape[:2]
    if len(image.shape) == 4:
        imsize = image.shape[1:3]
    cropping_center = np.clip(cropping_center, [target_height // 2, target_width // 2],
                              [imsize[0] - target_height // 2, imsize[1] - target_width // 2])

    if len(image.shape) == 3:
        cropped_image = image[cropping_center[0] - target_height // 2: cropping_center[0] + target_height // 2,
                        cropping_center[1] - target_width // 2: cropping_center[1] + target_width // 2]
    elif len(image.shape) == 4:
        cropped_image = image[:, cropping_center[0] - target_height // 2: cropping_center[0] + target_height // 2,
                        cropping_center[1] - target_width // 2: cropping_center[1] + target_width // 2]
    else:
        raise NotImplementedError
    return cropped_image


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

            for key in MANDATORY_KEYS:
                assert key in traj_meta_data, 'missing key {} in metadata!'.format(key)
            assert isinstance(traj_meta_data['object_classes'], list), "did not split object classes!"
            assert all(
                [isinstance(x, str) for x in traj_meta_data['object_classes']]), 'object classes is not a string!'
            trajs.append((t, traj_meta_data))
    return annotations_loaded

def run(args):
    if args.output_folder == '':
        args.output_folder = os.path.join(args.input_folder, 'lmdb')
    else:
        args.output_folder = os.path.join(args.output_folder, 'lmdb')

    args.input_folder, args.output_folder = [os.path.expanduser(x) for x in (args.input_folder, args.output_folder)]

    trajs, annotations_loaded = [], 0
    if os.path.isfile('{}/collection_metadata.json'.format(args.input_folder)):
        general_meta_data_dict = json.load(open('{}/collection_metadata.json'.format(args.input_folder), 'r'))
        print('found general metadatadict.')
    else:
        general_meta_data_dict = None

    datetime_groups = glob.glob(args.input_folder + "/*")
    datetime_groups = [dir for dir in datetime_groups if '/lmdb' not in dir]
    print('found {} datetime groups!'.format(len(datetime_groups)))
    print(datetime_groups)
    num_files = counting_files(datetime_groups)
    print('found {} files'.format(num_files))

    for dg in datetime_groups:
        print('processing datatime group', dg)
        traj_groups = glob.glob(dg + "/raw/traj_group*")
        print('found {} traj groups!'.format(len(traj_groups)))

        metadata_dict_datetimegroup = '{}/collection_metadata.json'.format(dg)
        if os.path.exists(metadata_dict_datetimegroup):  # try to load metadatadict insdie group if it's not provided
            general_meta_data_dict = json.load(open(metadata_dict_datetimegroup, 'r'))
        annotations_loaded = load_groups(trajs, traj_groups, annotations_loaded, general_meta_data_dict)

    random.shuffle(trajs)
    print('Loaded {} trajectories with {} annotations!'.format(len(trajs), annotations_loaded))
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
        if not args.overwrite_existing:
            try:
                if num_files == loading_num_files(args.output_folder):
                    print('skipping {} because file numbers match!'.format(args.output_folder))
                    return
            except:
                pass
        shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)
    print('saving outputs to', args.output_folder)

    random.shuffle(trajs)
    ntraj = len(trajs)
    phases = ['train', 'val', 'test']
    n_files_cum = np.cumsum((np.array(args.train_val_split) * ntraj).astype(np.int))
    train_trajs = trajs[0: n_files_cum[0]]
    val_trajs = trajs[n_files_cum[0]: n_files_cum[1]]
    test_trajs = trajs[n_files_cum[2]:]
    assert len(train_trajs) + len(val_trajs) + len(test_trajs) == len(trajs)

    for phase, phase_trajs in zip(phases, [train_trajs, val_trajs, test_trajs]):
        extract_trajs(args, phase, phase_trajs, num_files)
    if args.compute_normalization:
        compute_dataset_normalization(args.output_folder)

def loading_num_files(lmdb_folder):
    data_base = lmdb.open(os.path.join(lmdb_folder, 'train', 'lmdb.data'), subdir=False,
                                     readonly=True, lock=False,
                                     readahead=False, meminit=False)
    with data_base.begin(write=False) as txn:
        num_files = pa.deserialize(txn.get(b'num_files'))
    return num_files

def counting_files(dir_list):
    print('getting directory hash')
    overall_number = 0
    for dir in dir_list:
        cmd = 'find {} -type f | wc -l'.format(dir)
        import subprocess
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        overall_number += int(output)
    print('overall number of dirs', overall_number)
    return overall_number

def extract_trajs(args, phase, trajs, num_files):
    # print('############################o')
    # print('############################o')
    # print('############################o')
    # print('############################o')
    # trajs = trajs[:10]
    # print('only using {} trajs!!!!'.format(len(trajs)))

    output_folder = os.path.join(args.output_folder, phase)
    os.makedirs(output_folder)

    lmdb_path = os.path.join(output_folder, 'lmdb.data')
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=False,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    database_index = 0
    index_dict = {}
    meta_data_list = []
    for i, traj in enumerate(tqdm(trajs)):
        index_dict, database_index, txn, db, meta_data_list = save_worker(i, index_dict, traj, args, database_index, txn, db, meta_data_list)
    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(database_index + 1)]

    domain_hashes = set()
    for m in meta_data_list:
        domain_hashes.add(m['domain_hash'])

    print('found {} distinct domain hashes'.format(len(domain_hashes)))

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))
        txn.put(b'index_dict', dumps_pyarrow(index_dict))
        txn.put(b'meta_data_list', dumps_pyarrow(meta_data_list))
        txn.put(b'num_files', dumps_pyarrow(num_files))

    print("Flushing database ...")
    db.sync()
    db.close()

def parse_args():
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to lmdb")
    parser.add_argument('input_folder', type=str, help='where raw files are stored')
    parser.add_argument('--output_folder', type=str, help='where to save', default='')
    parser.add_argument('--n_workers', type=int, help='number of multi-threaded workers', default=1)
    parser.add_argument('--task_stage', type=int, default=-1, help='only save time-steps from the desired task stage')
    parser.add_argument('--down_sample', nargs='+', default=[112, 144], help='save resolution to use, otherwise take image orignal resolution', type=int)
    parser.add_argument('--train_val_split', nargs='+', default=[0.9, 0.1, 0], help='save trajs into seperate folders for train and val', type=float)
    parser.add_argument('--select_image_croppping_pos', action='store_true', help='save time steps separately')
    parser.add_argument('--compute_normalization', action='store_true', help='compute normalization')
    parser.add_argument('--crop_at_coords', nargs='+', default=[], help='crop at specified coordinates (height width)', type=int)
    parser.add_argument('--overwrite_existing', action='store_true', help='skip exisitng lmdb folders')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    run(parse_args())
