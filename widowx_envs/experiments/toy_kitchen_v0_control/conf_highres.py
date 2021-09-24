import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_15-33-28/raw/traj_group0/traj8', 0]

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'action_mode':'3trans3rot',
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
    # 'load_goal_image': [load_traj, 18],
}

policy = [
{
    'type': GCBCPolicyImages,

    # long traj task id conditioned
    # "human_demo, put potato on plate": 0,
    # "human_demo, put lemon on plate": 1,
    # "human_demo, put can in pot": 2,
    # "human_demo, put corn on plate": 3
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/long_traj_only_2021-07-08_18-39-18/weights/weights_best_itr97700.pth',

    'confirm_first_image': False,
    # 'crop_image_region': [31, 88],
    'model_override_params': {
        'data_conf': {
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128],
        'sel_camera': 0,
        'state_dim': 7,
        'test_time_task_id': 0,
    },
}
]


config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    'save_format': ['raw'],
}