import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
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

    # "human_demo, put knife in pot or pan": 0,
    # "human_demo, put red bottle in sink": 1,
    # "human_demo, put corn on plate": 2,
    # "human_demo, take sushi out of pan": 3,
    # "human_demo, put pear in bowl": 4,
    # "human_demo, put spoon on plate": 5,
    # "human_demo, put sweet_potato in pan which is on stove": 6,
    # "human_demo, put pear on plate": 7,
    # "human_demo, put eggplant into pot or pan": 8,
    # "human_demo, put spoon into pan": 9,
    # "human_demo, pick up pan from stove": 10,
    # "human_demo, put carrot on plate": 11,
    # "human_demo, flip salt upright": 12,
    # "human_demo, put spatula in pan": 13,
    # "human_demo, flip cup upright": 14,
    # "human_demo, pick up sponge and wipe plate": 15,
    # "human_demo, take lid off pot or pan": 16,
    # "human_demo, put potato in pot or pan": 17,
    # "human_demo, put eggplant on plate": 18,
    # "human_demo, turn faucet front to left (in the eyes of the robot)": 19,
    # "human_demo, put pepper in pot or pan": 20,
    # "human_demo, put pot or pan in sink": 21,
    # "human_demo, pick up glass cup": 22,
    # "human_demo, put spoon in pot": 23,
    # "human_demo, twist knob start vertical _clockwise90": 24,
    # "human_demo, put lemon on plate": 25,
    # "human_demo, put knife on cutting board": 26,
    # "human_demo, pick up green mug": 27,
    # "human_demo, put cup from anywhere into sink": 28,
    # "human_demo, put corn in pan which is on stove": 29,
    # "human_demo, put sushi in pot or pan": 30,
    # "human_demo, open box": 31,
    # "human_demo, put brush into pot or pan": 32,
    # "human_demo, put detergent in sink": 33,
    # "human_demo, put can in pot": 34,
    # "human_demo, put corn into bowl": 35,
    # "human_demo, put corn in pot which is in sink": 36,
    # "human_demo, put sushi on plate": 37,
    # "human_demo, put sweet potato in pot": 38,
    # "human_demo, flip orange pot upright in sink": 39,
    # "human_demo, put broccoli in pot or pan": 40,
    # "human_demo, put strawberry in pot": 41,
    # "human_demo, put cup into pot or pan": 42,
    # "human_demo, pick up any cup": 43,
    # "human_demo, turn lever vertical to front": 44,
    # "human_demo, flip pot upright which is in sink": 45,
    # "human_demo, put carrot in pot or pan": 46,
    # "human_demo, put carrot on cutting board": 47,
    # "human_demo, put carrot in bowl": 48,
    # "human_demo, put detergent from sink into drying rack": 49,
    # "human_demo, put potato on plate": 50,
    # "human_demo, lift bowl": 51,
    # "human_demo, pick up bowl and put in small 4-flap box": 52,
    # "human_demo, pick up pot from sink": 53,
    # "human_demo, take carrot off plate": 54,
    # "human_demo, put broccoli in bowl": 55,
    # "human_demo, take broccoli out of pan": 56,
    # "human_demo, put green squash into pot or pan": 57,
    # "human_demo, put pot or pan on stove": 58,
    # "human_demo, put lid on stove": 59,
    # "human_demo, close box": 60,
    # "human_demo, take can out of pan": 61,
    # "human_demo, put lid on pot or pan": 62,
    # "human_demo, put pot or pan from sink into drying rack": 63,
    # "human_demo, put banana on plate": 64,
    # "human_demo, put banana in pot or pan": 65

    'restore_path': '/home/dcuser1/experiments/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/multicam_2021-09-03_00-12-45/weights/weights_best_itr343616.pth',

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