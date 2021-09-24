import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'action_mode':'3trans3rot',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    # 'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/lever_vertical_to_front/2021-05-25_16-44-03/raw/traj_group0/traj12',
    # 'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/initial_testconfig/2021-06-02_14-02-06/raw/traj_group0/traj0',
    'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/turn_lever_vertical_to_front_distractors/2021-05-29_14-16-45/raw/traj_group0/traj0/',
    # cropped, success: 0, 3, 53
    # whole_large: fail 0, 3,  success 53
    # whole_small: fail 0, 3, 53
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 30,
    # 'image_height': 56,
    # 'image_width': 72,
    'image_height': 96,  # when doing cropping !!!!!!!!!!
    'image_width': 128, # when doing cropping !!!!!!!!!!
    # 'image_height': 112,  # for highres
    # 'image_width': 144,   # for highres
    'make_final_gif': True,
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    # 'load_goal_image': [os.environ['DATA'] + '/robonetv2/toykitchen_v0/front_to_left/2021-05-20_12-44-27/raw/traj_group0/traj0', 48],
}

from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy
from widowx_envs.policies.policy import NullPolicy
policy = [
{
    'type': GCBCPolicyImages,
    # BC fromscratch
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/cropped/weights_backup/weights_itr40660.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/whole_small/weights/weights_itr101244.pth'  ,
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/whole_large/weights/weights_itr101244.pth'  ,

    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/distractors_cropped96_b8_2021-06-01_04-01-44/weights/weights_itr163800.pth',

    'confirm_first_image': True,
    # 'crop_image_region': [31, 88],
    # 'crop_image_region': 'select',
    'model_override_params': {
        # 'data_conf': {
        #     'random_crop': [96, 128],
        #     'image_size_beforecrop': [112, 144]
        # },
        # 'img_sz': [96, 128],
        'data_conf': {
            'random_crop': [48, 64],
            'image_size_beforecrop': [56, 72]
        },
        'img_sz': [48, 64],
        'sel_camera': 0,
        'state_dim': 7,
    },
}
]


config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,  # by default
    'save_format': ['raw'],
    'make_diagnostics': True
}