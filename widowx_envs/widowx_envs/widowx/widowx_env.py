import rospy
from sensor_msgs.msg import Image
import pickle as pkl
import time
from widowx_envs.utils.transformation_utils import state2transform
import numpy as np
import math
from widowx_envs.base.base_env2 import BaseRobotEnv2
from widowx_envs.widowx.src.widowx_controller import WidowX_Controller
from widowx_envs.widowx.src.vr_controller_client import WidowX_VRContollerClient
from widowx_envs.utils.exceptions import Environment_Exception
from widowx_envs.utils.utils import ask_confirm
import os
from gym import spaces
import random


class WidowXEnv(BaseRobotEnv2):
  
    def _default_hparams(self):
        robot_name = os.getenv('ROBONETV2_ARM')
        if robot_name is None:
            print('Environment variable ROBONETV2_ARM has to be set. Please define it based on https://github.com/Interbotix/interbotix_ros_manipulators/tree/main/interbotix_ros_xsarms')
            print('For instance in case of WidowX 250 Robot Arm 6DOF use:')
            print('echo "export ROBONETV2_ARM=wx250s" >> ~/.bashrc && source ~/.bashrc')
            raise RuntimeError
        default_dict = {
            'robot_name': robot_name,
            'randomize_initpos': 'full_area',
            'mode_rel': [True, True, True, True, True],
            'start_state': None,
            'start_transform': None,
            'skip_move_to_neutral': False,
            'move_to_rand_start_freq': 1,
            'fix_zangle': False
        }
        parent_params = super(WidowXEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None, reset_state=None):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        self._controller.check_motor_status_and_reboot()
        self._controller.open_gripper(True)
        if not self._hp.skip_move_to_neutral:
            self._controller.move_to_neutral(duration=1.5)

        if self._hp.move_to_rand_start_freq != -1:
            if itraj % self._hp.move_to_rand_start_freq == 0:
                self.move_to_startstate()

        self._reset_previous_qpos()
        return self._get_obs()

    def move_to_startstate(self):
        if self._hp.start_state is not None:
            start_state = pkl.load(open(self._hp.start_state + '/obs_dict.pkl', 'rb'))['state'][0]
            if start_state.shape[0] == 5:
                start_state = np.concatenate([start_state[:3], np.zeros(2), start_state[3:]])
            transform, _ = state2transform(start_state, self._controller.default_rot)
            assert isinstance(self._controller, WidowX_Controller)
            self._controller.move_to_eep(transform)
        if self._hp.start_transform is not None:
            path = self._hp.start_transform[0]
            tstep = self._hp.start_transform[1]
            transform = pkl.load(open(path + '/obs_dict.pkl', 'rb'))['eef_transform'][tstep]
            assert isinstance(self._controller, WidowX_Controller)
            self._controller.move_to_eep(transform)
        else:
            if self._hp.randomize_initpos == 'restricted_space':
                lower = np.array([0.14, 0, 0.05])
                upper = np.array([0.21, 0.14, 0.14])
                startpos = np.random.uniform(lower, upper)
                # zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
                zangle = np.random.uniform(0.5*np.pi, -0.5*np.pi)  # np.pi is neutral!
            elif self._hp.randomize_initpos == 'full_area':
                startpos = np.random.uniform(self._low_bound[:3] + np.array([0, 0, 0.05]), self._high_bound[:3])
                zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
            elif self._hp.randomize_initpos == 'line':
                y_rand = np.random.uniform(self._low_bound[1] + 0.05, self._high_bound[1] - 0.04)
                z_rand = np.random.uniform(self._low_bound[2] + 0.05, self._high_bound[2])
                startpos = np.array([0.21, y_rand, z_rand])
                zangle = math.pi / 2
            else:
                raise NotImplementedError
            try:
                if self._hp.fix_zangle:
                    zangle = 0
                self._controller.move_to_state(startpos, zangle, duration=1.5)
            except Environment_Exception:
                self.move_to_startstate()  # retry with different sample position

    def start(self):
        self._controller.set_moving_time(self._hp.move_duration)

    def finish(self):
        pass

    def ask_confirmation(self, ):
        return ask_confirm("Was the trajectory okay? y/n")


class RandomInit_WidowXEnv(WidowXEnv):
    def move_to_startstate(self):
        assert self._hp.start_transform is not None
        sampled_start_transform = random.choice(self._hp.start_transform)
        path = sampled_start_transform[0]
        tstep = sampled_start_transform[1]
        obs_dict = pkl.load(open(path + '/obs_dict.pkl', 'rb'))
        assert isinstance(self._controller, WidowX_Controller)
        # transform = obs_dict['eef_transform'][tstep]
        # self._controller.move_to_eep(transform)
        joint_angles = obs_dict['qpos'][tstep]
        self._controller.bot.arm.publish_positions(joint_angles, moving_time=1.5)
        # self._controller.wait_until_gripper_position_reached()
        # while self._controller.get_continuous_gripper_position() < 0.98:
        #     print(self._controller.get_continuous_gripper_position())
        #     self._controller._gripper.open()

class VR_WidowX(WidowXEnv):
    def __init__(self, env_params=None):
        super(VR_WidowX, self).__init__(env_params)
        self.task_stage = 0

        from oculus_reader import OculusReader
        self.oculus_reader = OculusReader()

    def get_vr_buttons(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'RG' in buttons:
            buttons['handle'] = buttons['RG']
        else:
            buttons['handle'] = False
        return buttons

    def _default_hparams(self):
        default_dict = {
            'num_task_stages': 1,
            'make_oculus_reader': True

        }
        parent_params = super(VR_WidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def step(self, action, get_obs_tstamp=None, blocking=False):
        """
        :param action:  endeffector velocities
        :return:  observations
        """
        obs = super(VR_WidowX, self).step(action, get_obs_tstamp, blocking)
        if self.get_vr_buttons()['B']:
            self.task_stage += 1
        return obs

    def _get_obs(self):
        obs = super(VR_WidowX, self)._get_obs()
        if self.task_stage == self._hp.num_task_stages:
            obs['env_done'] = True
        obs['task_stage'] = self.task_stage
        print("task stage", obs['task_stage'])
        return obs

    def reset(self, itraj=None, reset_state=None):
        self.task_stage = 0
        obs = super(VR_WidowX, self).reset(itraj=itraj)
        start_key = 'handle'
        print('waiting for {} button press to start recording. Press B to go to neutral.'.format(start_key))
        buttons = self.get_vr_buttons()
        while not buttons[start_key]:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)
            if 'B' in buttons and buttons['B']:
                self.move_to_neutral()
                print("moved to neutral. waiting for {} button press to start recording.".format(start_key))
        return self._get_obs()

    def ask_confirmation(self):
        print('current endeffector pos', self.get_full_state()[:3])
        print('current joint angles pos', self._controller.get_joint_angles())
        print('Was the trajectory okay? Press A to confirm and RJ to discard')
        buttons = self.get_vr_buttons()
        while not buttons['A'] and not buttons['RJ']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        if buttons['A']:
            print('trajectory accepted!')
            return True

class VR_WidowX_DAgger(VR_WidowX):
    def __init__(self, env_params=None):
        super(VR_WidowX_DAgger, self).__init__(env_params)

    def ask_confirmation(self):
        print('Was the trajectory okay? Press A to save a successful trajectory, trigger to save an unsuccessful trajectory, and RJ to discard')
        buttons = self.get_vr_buttons()

        while buttons['A'] or buttons['RJ'] or buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        while not buttons['A'] and not buttons['RJ'] and not buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        elif buttons['A']:
            print('successful trajectory accepted!')
            return 'Success'
        elif buttons['RTr']:
            print('unsuccessful trajectory accepted!')
            return 'Failure'

class VR_WidowX_ClientServer(WidowXEnv):
    def __init__(self, env_params=None):
        super(VR_WidowX_ClientServer, self).__init__(env_params)
        assert isinstance(self._controller, WidowX_VRContollerClient)

        self.task_stage = 0
        # self._controller.num_task_stages = self._hp.num_task_stages

    def _default_hparams(self):
        default_dict = {
            'num_task_stages': 1
        }
        parent_params = super(VR_WidowX_ClientServer, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def step(self, action):
        """
        :param action:  endeffector velocities
        :return:  observations
        """
        t1 = time.time()
        if self._controller.get_vr_buttons()['B']:
            self.task_stage += 1
        obs = self._get_obs()
        print('time to get obs', time.time() - t1)
        return obs

    def _get_obs(self):
        obs = super(VR_WidowX_ClientServer, self)._get_obs()
        if self.task_stage == self._hp.num_task_stages:
            obs['env_done'] = True
        obs['task_stage'] = self.task_stage
        print("task stage", obs['task_stage'])
        return obs

    def reset(self, itraj=None, reset_state=None):
        self.task_stage = 0
        obs = super(VR_WidowX_ClientServer, self).reset(itraj=itraj)
        start_key = 'handle'
        print('waiting for {} button press to start recording. Press B to stop recording.'.format(start_key))
        handle_button = self._controller.get_vr_buttons()[start_key]
        while not handle_button:
            handle_button = self._controller.get_vr_buttons()[start_key]
            rospy.sleep(0.01)
        obs = super(VR_WidowX_ClientServer, self).reset(itraj=itraj)
        return obs

    def ask_confirmation(self):
        print('Was the trajectory okay? Press A to confirm and RJ to discard')
        buttons = self._controller.get_vr_buttons()
        while not buttons['A'] and not buttons['RJ']:
            buttons = self._controller.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        if buttons['A']:
            print('trajectory accepted!')
            return True

    def start(self):
        self._controller.enable_controller()

    def finish(self):
        self._controller.disable_controller()


class StateReachingWidowX(WidowXEnv):
    def __init__(self, env_params=None, fixed_goal=False):
        super(StateReachingWidowX, self).__init__(env_params)

        action_space = np.full(self.adim, 0.05)
        self.action_space = spaces.Box(-action_space, action_space, dtype=np.float64)
        self.observation_space = spaces.dict.Dict({
            "state": spaces.Box(-np.full(7, np.inf), np.full(7, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
            "vector_to_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })
        self.goal_threshold = 0.0175
        self.fixed_goal = fixed_goal
        self.goal_coord = None

    def _default_hparams(self):
        default_dict = {
            'camera_topics': [],
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'action_space': '3trans',
            'fix_zangle': True,
            'override_workspace_boundaries': [[0.19, -0.08, 0.05, -1.57, 0], [0.31, 0.08, 0.055,  1.57, 0]]
        }
        parent_params = super(StateReachingWidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def _get_obs(self):
        full_obs = super(StateReachingWidowX, self)._get_obs()
        ee_coord = full_obs['full_state'][:3]
        vector_to_goal = self.goal_coord - ee_coord
        obs = {'vector_to_goal': vector_to_goal, 'state': self.get_full_state(),
               'joints': full_obs['qpos'], 'desired_goal': self.goal_coord, 'achieved_goal': ee_coord}
        return obs

    def step(self, action):
        obs = super(StateReachingWidowX, self).step(action)
        dist = np.linalg.norm(obs['vector_to_goal'])
        reward = -dist
        goal_reached = self.goal_reached(dist)
        return obs, reward, goal_reached, {}

    def goal_reached(self, distance):
        return distance < self.goal_threshold

    def reset(self):
        if self.fixed_goal:
            xyz = self._high_bound[:3] + self._low_bound[:3]
            xyz = xyz/2
        else:
            xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
        self.goal_coord = xyz
        obs = super(StateReachingWidowX, self).reset(itraj=0)
        return obs


class ImageReachingWidowX(StateReachingWidowX):
    def __init__(self, env_params=None, publish_images=True, fixed_image_size=64, fixed_goal=False):

        super(ImageReachingWidowX, self).__init__(env_params, fixed_goal=fixed_goal)
        self.image_size = fixed_image_size

        action_space = np.full(self.adim, 0.05)
        self.action_space = spaces.Box(-action_space, action_space, dtype=np.float64)
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(low=np.array([0]*self.image_size*self.image_size*3),
                                high=np.array([255]*self.image_size*self.image_size*3), dtype=np.uint8),
            "state": spaces.Box(-np.full(self.sdim, np.inf), np.full(self.sdim, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })
        self.goal_coord = None
        self.goal_threshold = 0.02
        self.publish_images = publish_images
        if self.publish_images:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher("/robonetv2_image/image_raw", Image, queue_size=10)

    def _default_hparams(self):
        from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
        default_dict = {
            'camera_topics': [IMTopic('/camera0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'transpose_image_to_chw': False,  # changes image to CHW format for use with pytorch
            'return_full_image': False,
        }
        parent_params = super(ImageReachingWidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    @staticmethod
    def _to_float32_flat_image(image):
        return np.float32(image.flatten()) / 255.0

    def _get_processed_image(self, image=None):
        if image is None:
            image = super(StateReachingWidowX, self)._get_obs()['images'][0]

        if self._hp['image_crop_xywh'] is None:
            trimmed_image = image
        else:
            x, y, w, h = self._hp['image_crop_xywh']
            trimmed_image = image[x:x+w, y:y+h]

        from skimage.transform import resize
        downsampled_trimmed_image = resize(trimmed_image, (self.image_size, self.image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        if self._hp['transpose_image_to_chw']:
            downsampled_trimmed_image = np.transpose(downsampled_trimmed_image, (2, 0, 1))
        return self._to_float32_flat_image(downsampled_trimmed_image)

    def step(self, action):
        obs = super(StateReachingWidowX, self).step(action)
        try:
            vector_to_goal = obs['vector_to_goal']
        except:
            raise NotImplementedError('Implemented only for the dict observation space')
        dist = np.linalg.norm(vector_to_goal)
        reward = -dist
        goal_reached = self.goal_reached(dist)
        if goal_reached:
            print('Done!')
        if self.publish_images:
            # try:
            self._publish_image(obs['image'])
            # except:
            #     raise NotImplementedError('Implemented only for the dict observation space')
        return obs, reward, goal_reached, {}

    def _publish_image(self, image):
        try:
            if self._hp['transpose_image_to_chw']:
                cv_image = np.uint8(image.reshape((3, self.image_size, self.image_size))*255.0)
                cv_image = np.transpose(cv_image, (1, 2, 0))
            else:
                cv_image = np.uint8(image.reshape((self.image_size, self.image_size, 3))*255.0)
            imgmsg = self.bridge.cv2_to_imgmsg(cv_image, 'rgb8')
            self.image_pub.publish(imgmsg)
        except Exception as e:
            print(e)

    def _get_obs(self):
        full_obs = super(StateReachingWidowX, self)._get_obs()
        image = full_obs['images'][0]
        ee_coord = full_obs['full_state'][:3]
        vector_to_goal = self.goal_coord - ee_coord
        processed_image = self._get_processed_image(image)

        obs = {'image': processed_image, 'state': self.get_full_state(), 'vector_to_goal': vector_to_goal,
               'joints': full_obs['qpos'], 'desired_goal': self.goal_coord, 'achieved_goal': ee_coord}
        if self._hp.return_full_image:
            obs['full_image'] = image
        # obs['gripper'] = full_obs['state'][-1]  # this dimension is not being updated for now
        return obs


if __name__ == '__main__':
    env = StateReachingWidowX()
    env.move_to_neutral()
    for i in range(5):
        env.reset()
        env.step(np.zeros(env.action_space.shape))
