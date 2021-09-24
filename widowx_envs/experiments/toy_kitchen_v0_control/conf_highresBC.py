import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop
from semiparametrictransfer.policies.rl_policy_bc import RLPolicyBC

# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/2021-08-27_14-17-17/raw/traj_group0/traj0', 169]
load_traj = ['/home/dcuser1/trainingdata/robonetv2/toykitchen_fixed_cam/toykitchen1/put_sweet_potato_in_pot_which_is_in_sink_distractors/2021-06-04_18-14-10/raw/traj_group0/traj14', 1]

env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
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
    'make_final_gif': True,
    'recreate_env': (False, 1),
    'ask_confirmation': False,
}

policy = {
    'type': RLPolicyBC,
    'log': True,
    'path': '/home/dcuser1/experiments/railrl_experiments/bc-resnet-test-exclude-deterministic/bc_resnet_test_exclude_deterministic_2021_08_26_06_39_49_0000--s-0/itr_190.pt',
    'exp_conf_path': '/home/dcuser1/experiments/railrl_experiments/bc-resnet-test-exclude-deterministic/bc_resnet_test_exclude_deterministic_2021_08_26_06_39_49_0000--s-0/variant.json',
    'num_tasks': 54,
    'task_id': 9,
    'confirm_first_image': True,
    'resnet': True,
    'true_normalize': True
}

config = {
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': True,  # by default
    'save_format': ['raw'],
}
