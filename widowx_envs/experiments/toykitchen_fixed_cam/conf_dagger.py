""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import VR_WidowX, VR_WidowX_DAgger
from widowx_envs.control_loops import TimedLoop
from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy, VRTeleopPolicyDAgger
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImagesOnline

env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'move_to_rand_start_freq': -1,
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'adaptive_wait': True,
    'override_workspace_boundaries': [[0.198, -0.038, 0.026, -1.57, 0], [0.320, 0.034, 0.1,  1.57, 0]],
    'action_clipping': None
}

agent = {
    'type': TimedLoop,
    'env': (VR_WidowX_DAgger, env_params),
    'recreate_env': (False, 1),
    'T': 500,
    'image_height': 480,
    'image_width': 640,
    'make_final_gif': False,
    'video_format': 'mp4',
}

policy = {
    'type': VRTeleopPolicyDAgger,
    'default_policy_type': GCBCPolicyImagesOnline, 
    'policy_T': 50,
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/random_mixing_kitchen1_2021-08-15_20-04-45/weights/weights_best_itr119510.pth', # TODO: change this path
    # 'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128],
        'sel_camera': 0,
        'state_dim': 7,
        'test_time_task_id': [8, 25, 45, 47, 53, 55], # TODO: change this to correct task id
        # Broccoli, Pan, lid, carrot, plate
    },
}

config = {
    'current_dir' : current_dir,
    'collection_metadata' : current_dir + '/collection_metadata.json',
    'start_index': 0,
    'end_index': 500,
    'agent': agent,
    'policy': policy,
    # 'save_data': True, # By default
    'save_format': ['raw'],
    'make_diagnostics': True
}


# "human_demo, pick up box cutter and put into drawer": 0,
# "human_demo, put fork from basket to tray": 1,
# "human_demo, put lid on stove": 2,
# "human_demo, flip pot upright which is in sink": 3,
# "human_demo, take sushi out of pan": 4,
# "human_demo, put eggplant into pot or pan": 5,
# "human_demo, put cup into pot or pan": 6,
# "human_demo, put potato in pot or pan": 7,
# "human_demo, take lid off pot or pan": 8,
# "human_demo, put big spoon from basket to tray": 9,
# "human_demo, put carrot in pot or pan": 10,
# "human_demo, put corn in pan which is on stove": 11,
# "human_demo, pick up violet Allen key": 12,
# "human_demo, take can out of pan": 13,
# "human_demo, put spatula in pan": 14,
# "human_demo, put brush into pot or pan": 15,
# "human_demo, put spoon in pot": 16,
# "human_demo, put pear in bowl": 17,
# "human_demo, put knife in pot or pan": 18,
# "human_demo, put detergent from sink into drying rack": 19,
# "human_demo, put can in pot": 20,
# "human_demo, put carrot on cutting board": 21,
# "human_demo, put banana on plate": 22,
# "human_demo, lift bowl": 23,
# "human_demo, twist knob start vertical _clockwise90": 24,
# "human_demo, put lid on pot or pan": 25,
# "human_demo, pick up red screwdriver": 26,
# "human_demo, put pot or pan in sink": 27,
# "human_demo, pick up pot from sink": 28,
# "human_demo, put pepper in pot or pan": 29,
# "human_demo, pick up pan from stove": 30,
# "human_demo, put strawberry in pot": 31,
# "human_demo, put broccoli in bowl": 32,
# "human_demo, put potato on plate": 33,
# "human_demo, put small spoon from basket to tray": 34,
# "human_demo, put pepper in pan": 35,
# "human_demo, put eggplant on plate": 36,
# "human_demo, pick up scissors and put into drawer": 37,
# "human_demo, put pot or pan from sink into drying rack": 38,
# "human_demo, put corn into bowl": 39,
# "human_demo, flip cup upright": 40,
# "human_demo, pick up glue and put into drawer": 41,
# "human_demo, pick up closest rainbow Allen key set": 42,
# "human_demo, put sweet potato in pot": 43,
# "human_demo, put spoon into pan": 44,
# "human_demo, take carrot off plate": 45,
# "human_demo, flip orange pot upright in sink": 46,
# "human_demo, take broccoli out of pan": 47,
# "human_demo, flip salt upright": 48,
# "human_demo, put sushi on plate": 49,
# "human_demo, put cup from anywhere into sink": 50,
# "human_demo, turn faucet front to left (in the eyes of the robot)": 51,
# "human_demo, put green squash into pot or pan": 52,
# "human_demo, put broccoli in pot or pan": 53,
# "human_demo, put sweet_potato in pan which is on stove": 54,
# "human_demo, put carrot on plate": 55,
# "human_demo, turn lever vertical to front": 56,
# "human_demo, put corn in pot which is in sink": 57,
# "human_demo, pick up bit holder": 58,
# "human_demo, pick up blue pen and put into drawer": 59,
# "human_demo, put knife on cutting board": 60,
# "human_demo, put red bottle in sink": 61,
# "human_demo, put pot or pan on stove": 62,
# "human_demo, put corn on plate": 63,
# "human_demo, put lemon on plate": 64,
# "human_demo, put detergent in sink": 65