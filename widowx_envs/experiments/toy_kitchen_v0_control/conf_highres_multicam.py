import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from imitation_learning.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/2021-08-27_14-19-53/raw/traj_group0/traj0', 55]

env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    'start_transform': load_traj,
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 50,
    'image_height': 480,  # for highres
    'image_width': 640,   # for highres
    'make_final_gif': False,
    'recreate_env': (False, 1),
    'ask_confirmation': False,
}

policy = [
{
    'type': GCBCPolicyImages,

    'restore_path': None, # Add your path to trained model checkpoint here

    'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128],
        'state_dim': 7,
        'test_time_task_id': 55,
    },
}
]


config = {
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': True,  # by default
    'save_format': ['raw'],
}