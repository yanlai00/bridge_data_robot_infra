""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import VR_WidowX
from widowx_envs.control_loops import TimedLoop
from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/camera1/color/image_raw'), IMTopic('/camera2/color/image_raw'), IMTopic('/camera3/color/image_raw')],
    'depth_camera_topics': [IMTopic('/camera1/depth/image_rect_raw', dtype='16UC1'), IMTopic('/camera2/depth/image_rect_raw', dtype='16UC1'), IMTopic('/camera3/depth/image_rect_raw', dtype='16UC1')],
    #'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    # 'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'action_space':'3trans',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.198, -0.038, 0.026, -1.57, 0], [0.320, 0.034, 0.1,  1.57, 0]],
    # 'action_clipping': None
}

agent = {
    'type': TimedLoop,
    'env': (VR_WidowX, env_params),
    'recreate_env': (False, 1),
    'T': 100,
    # 'image_height': 480,
    # 'image_width': 640,
    'image_height': 96,
    'image_width': 128,
    'make_final_gif': False,
    'video_format': 'mp4',
}

policy = {
    'type': VRTeleopPolicy,
}

config = {
    'current_dir' : current_dir,
    'collection_metadata' : current_dir + '/collection_metadata.json',
    'start_index':0,
    'end_index': 500,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,
    'save_format': ['raw'],
}
