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
from widowx_envs.widowx.widowx_env import WidowXEnv
import os
from gym import spaces


class WidowxImageTimedEnv(WidowXEnv):

    def __init__(self, env_params=None, publish_images=False, image_height=48, image_width=64):

        super(WidowxImageTimedEnv, self).__init__(env_params)
        self.image_height = image_height
        self.image_width = image_width

        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(low=np.array([0]*self.image_height*self.image_width*3),
                                high=np.array([255]*self.image_height*self.image_width*3), dtype=np.uint8),
            # "state": spaces.Box(-np.full(self.sdim, np.inf), np.full(self.sdim, np.inf), dtype=np.float64),
        })
        self.publish_images = publish_images
        self.step_duration = self._hp.move_duration
        self.start()
        if self.publish_images:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher("/robonetv2_image/image_raw", Image, queue_size=10)
    
    def reset(self, itraj=0, reset_state=None):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        return super(WidowxImageTimedEnv, self).reset(itraj=itraj, reset_state=reset_state)

    def _default_hparams(self):
        from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
        default_dict = {
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1, 1.57, 0]],
            'action_clipping': None,
            'move_duration': 0.2,
            'fix_zangle': 0.1,
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'transpose_image_to_chw': False,  # changes image to CHW format for use with pytorch
            'return_full_image': False,
        }
        parent_params = super(WidowxImageTimedEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    @staticmethod
    def _to_float32_flat_image(image):
        return np.float32(image.flatten()) / 255.0

    def _get_processed_image(self, image=None):
        if image is None:
            image = super(WidowxImageTimedEnv, self)._get_obs()['images'][0]

        if self._hp['image_crop_xywh'] is None:
            trimmed_image = image
        else:
            x, y, w, h = self._hp['image_crop_xywh']
            trimmed_image = image[x:x+w, y:y+h]

        from skimage.transform import resize
        downsampled_trimmed_image = resize(trimmed_image, (self.image_height, self.image_width), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        if self._hp['transpose_image_to_chw']:
            downsampled_trimmed_image = np.transpose(downsampled_trimmed_image, (2, 0, 1))
        return self._to_float32_flat_image(downsampled_trimmed_image)

    def step(self, action, blocking=False):
        last_tstep = time.time()
        obs = super(WidowxImageTimedEnv, self).step(action, blocking=blocking)
        goal_reached = self.goal_reached(obs)
        if self.publish_images:
            self._publish_image(obs['image'])
        if (time.time() - last_tstep) > self.step_duration * 1.05:
            print('###########################')
            print('Warning, loop takes too long: {}s!!!'.format(time.time() - last_tstep))
            print('###########################')
        
        # while True:
        #     if (time.time() - last_tstep) > self.step_duration:
        #         if (time.time() - last_tstep) > self.step_duration*1.05:
        #             print('###########################')
        #             print('Warning, loop takes too long: {}s!!!'.format(time.time() - last_tstep))
        #             print('###########################')
        #         break

        return obs, int(goal_reached), goal_reached, {}

    def _publish_image(self, image):
        try:
            if self._hp['transpose_image_to_chw']:
                cv_image = np.uint8(image.reshape((3, self.image_height, self.image_width))*255.0)
                cv_image = np.transpose(cv_image, (1, 2, 0))
            else:
                cv_image = np.uint8(image.reshape((self.image_height, self.image_width, 3))*255.0)
            imgmsg = self.bridge.cv2_to_imgmsg(cv_image, 'rgb8')
            self.image_pub.publish(imgmsg)
        except Exception as e:
            print(e)

    def _get_obs(self):
        full_obs = super(WidowxImageTimedEnv, self)._get_obs()
        image = full_obs['images'][0]
        ee_coord = full_obs['full_state'][:3]
        processed_image = self._get_processed_image(image)

        obs = {'image': processed_image, 'state': self.get_full_state(),
               'joints': full_obs['qpos'], 'achieved_goal': ee_coord}
        if self._hp.return_full_image:
            obs['full_image'] = image
        return obs
    
    def goal_reached(self, obs):
        # TODO: incorporate classifier here
        return False


if __name__ == '__main__':
    env = WidowxImageTimedEnv()
    env.move_to_neutral()
    for i in range(2):
        env.reset()
        r = rospy.Rate(5)
        start = time.time()
        for j in range(10):
            env.step(np.zeros(env.action_space.shape))
            r.sleep()
        end = time.time()
        print("time", end - start)
