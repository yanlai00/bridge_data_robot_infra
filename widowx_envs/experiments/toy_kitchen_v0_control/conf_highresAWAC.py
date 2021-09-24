import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop
from semiparametrictransfer.policies.rl_policy_awac import RLPolicyAWAC

# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_carrot_on_cutting_board/2021-06-08_18-42-42/raw/traj_group0/traj0', 0]
load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_sweet_potato_in_pot_which_is_in_sink_distractors/2021-06-03_16-50-31/raw/traj_group0/traj0', 0]

env_params = {
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
    'make_final_gif': True,
    'recreate_env': (False, 1),
    'ask_confirmation': False,
}

policy = {
    'type': RLPolicyAWAC,
    'log': True,
    # 'path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-bc/kitchen1_bc_2021_08_05_01_34_32_0000--s-0/model_pkl/290.pt',
    'path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-test-awac/kitchen1_test_awac_2021_08_05_19_16_37_0000--s-0/model_pkl/160.pt',
    'hist': False,
    'optimize_q_function': False,
    'vqvae': False,
    # 'exp_conf_path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-bc/kitchen1_bc_2021_08_05_01_34_32_0000--s-0/variant.json',
    'exp_conf_path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-test-awac/kitchen1_test_awac_2021_08_05_19_16_37_0000--s-0/variant.json',
    'num_tasks': 12,
    'task_id': 10,
    'confirm_first_image': True,
}

config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': True,  # by default
    'save_format': ['raw'],
}
