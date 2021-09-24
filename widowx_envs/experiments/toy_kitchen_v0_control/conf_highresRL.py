import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop
from semiparametrictransfer.policies.rl_policy_cog import RLPolicyCOG

load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_sweet_potato_in_pot/2021-06-11_17-31-39/raw/traj_group0/traj3', 3]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_16-42-48/raw/traj_group0/traj0', 0]

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
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
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    # 'load_goal_image': [load_traj, 18],
}

from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy
from widowx_envs.policies.policy import NullPolicy

policy = {
    'type': RLPolicyCOG,
    'log': True,
    # 'path' : '/home/datacol1/anikait_eval/bc/40.pt',
    # 'path': '/home/datacol1/jul18/rescale-mse-kitchen-bc/rescale_mse_kitchen_bc_2021_07_17_16_38_15_0000--s-0/model_pkl/60.pt',
    # 'path': '/home/datacol1/jul18/rescale-kitchen-bc-vqvae/rescale_kitchen_bc_vqvae_2021_07_17_16_38_03_0000--s-0/model_pkl/40.pt',
    # 'path': '/home/datacol1/jul18/pes0-rescale-randview-kitchen-minq0-task1/pes0_rescale_randview_kitchen_minq0_task1_2021_07_17_16_38_32_0000--s-0/model_pkl/105.pt',
    # 'path': '/home/datacol1/jul18/rescale-mse-kitchen-bc-vqvae/rescale_mse_kitchen_bc_vqvae_2021_07_17_16_38_11_0000--s-0/model_pkl/40.pt',
    # 'path': '/home/datacol1/jul18/pes0-rescale-randview-kitchen-minq1-task1/pes0_rescale_randview_kitchen_minq1_task1_2021_07_17_16_38_27_0000--s-0/model_pkl/100.pt',
    # 'path': '/home/datacol1/jul18/pes0-rescale-randview-kitchen-minq.5-task1/pes0_rescale_randview_kitchen_minq.5_task1_2021_07_17_16_38_23_0000--s-0/model_pkl/100.pt',
    # 'path': '/home/datacol1/jul18/pes0-rescale-randview-kitchen-minq1-tasksingle/pes0_rescale_randview_kitchen_minq1_tasksingle_2021_07_17_16_38_21_0000--s-0/model_pkl/100.pt',
    # 'path' : '/home/datacol1/anikait_eval/jul21/nonzereopes_rescale_randview_kitchen_minq1_tasksingle_resnet_2021_07_20_00_44_14_0000--s-0/model_pkl/80.pt',
    # 'path' : '/home/datacol1/anikait_eval/jul21/pes0-rescale-randview-kitchen-minq1-tasksingle-dr30.1/pes0_rescale_randview_kitchen_minq1_tasksingle_dr30.1_2021_07_20_00_44_19_0000--s-0/model_pkl/60.pt',    
    # 'path' : '/home/datacol1/anikait_eval/jul21/pes0-rescale-randview-kitchen-minq1-tasksingle-dr30.01/pes0_rescale_randview_kitchen_minq1_tasksingle_dr30.01_2021_07_20_00_44_28_0000--s-0/model_pkl/135.pt',
    'path' : '/home/datacol1/400.pt',    
    'policy_type': 1,
    # 'history': True,
    # 'optimize_q_function': False,
    # 'bottleneck': True,
    # 'vqvae': False,
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