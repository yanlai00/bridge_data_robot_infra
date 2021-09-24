""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop
from widowx_envs.policies.test_latency_policy import TestLatency

env_params = {
    # 'camera_topics': [IMTopic('/camera0/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
    # 'camera_topics': [IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    # 'depth_camera_topics': [IMTopic('/camera0/depth/image_rect_raw', dtype='16UC1')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'move_to_rand_start_freq': 2,
    'action_mode':'3trans',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'adaptive_wait': True,
    # 'wait_time': 0.15,
    # 'action_clipping': None,
    # 'override_workspace_boundaries': [[0.19, -0.14, 0.03, -1.57, 0], [0.38, 0.14, 0.12, 1.57, 0]]
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]]
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'recreate_env': (False, 1),
    'T': 13,
    # 'image_height': 480,
    # 'image_width': 640,
    'image_height': 96,
    'image_width': 128,
    'make_final_gif': False,
    'video_format': 'mp4',
    'ask_confirmation':False,
}

policy = {
    'type': TestLatency,
}

config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 500,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,
    'save_format': ['raw'],
    'make_diagnostics':True
}
