from widowx_envs.policies.policy import Policy
from widowx_envs.utils.utils import AttrDict
from widowx_envs.control_loops import Environment_Exception
import widowx_envs.utils.transformation_utils as tr
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages

import numpy as np
import time

from pyquaternion import Quaternion
from transformations import quaternion_from_matrix

import rospy
import tf2_ros
import geometry_msgs.msg
import random

def publish_transform(transform, name):
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'wx250s/base_link'
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.w = quat[0]
    t.transform.rotation.x = quat[1]
    t.transform.rotation.y = quat[2]
    t.transform.rotation.z = quat[3]

    # print('publish transofrm', name)
    br.sendTransform(t)

class VRTeleopPolicy(Policy):
    def __init__(self, ag_params, policyparams):

        """ Computes actions from states/observations. """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.last_pressed_times = {}
        self.env = ag_params.env_handle

        self.reader = self.env.oculus_reader
        # self.prev_vr_transform = None
        self.action_space = self.env._hp.action_mode

        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None
        self.internal_counter = 0
        self.internal_counter_default_policy = 0

    def _default_hparams(self):
        dict = AttrDict(
            load_file="",
            type=None,
            policy_T=None,
        )
        default_dict = super(Policy, self)._default_hparams()
        default_dict.update(dict)
        return default_dict


    def get_pose_and_button(self):
        poses, buttons = self.reader.get_transformations_and_buttons()
        if poses == {}:
            return None, None, None, None
        return poses['r'], buttons['RTr'], buttons['rightTrig'][0], buttons['RG']


    def act_use_fixed_reference(self, t, i_tr, images):
        # print("time update cmds", time.time() - self.last_update_time)
        self.last_update_time = time.time()
        t1 = time.time()
        current_vr_transform, trigger, trigger_continuous, handle_button = self.get_pose_and_button()
        if current_vr_transform is None:
            return self.get_default_action(t, i_tr, images)
        else:
            if not self.prev_handle_press and handle_button:
                print("resetting reference pose")
                self.internal_counter_default_policy = 0
                self.reference_vr_transform = self.oculus_to_robot(current_vr_transform)
                self.initial_vr_offset = tr.RpToTrans(np.eye(3), self.reference_vr_transform[:3, 3])
                self.reference_vr_transform = tr.TransInv(self.initial_vr_offset).dot(self.reference_vr_transform)  ##

                self.reference_robot_transform, _ = self.env.get_target_state()
                if self.action_space == '3trans1rot':
                    self.reference_robot_transform = self.zero_out_pitchroll(self.reference_robot_transform)
                self.prev_commanded_transform = self.reference_robot_transform

            if not handle_button:
                self.internal_counter = 0
                self.internal_counter_default_policy += 1
                self.reference_vr_transform = None
                self.reference_robot_transform, _ = self.env.get_target_state()
                self.prev_handle_press = False
                if self.action_space == '3trans1rot':
                    self.reference_robot_transform = self.zero_out_pitchroll(self.reference_robot_transform)
                self.prev_commanded_transform = self.reference_robot_transform
                return self.get_default_action(t, i_tr, images)
        self.prev_handle_press = True
        self.internal_counter += 1

        current_vr_transform = self.oculus_to_robot(current_vr_transform)
        current_vr_transform = tr.TransInv(self.initial_vr_offset).dot(current_vr_transform)  ##

        publish_transform(current_vr_transform, 'currentvr_robotsystem')
        delta_vr_transform = current_vr_transform.dot(tr.TransInv(self.reference_vr_transform))

        publish_transform(self.reference_robot_transform, 'reference_robot_transform')
        M_rob, p_rob = tr.TransToRp(self.reference_robot_transform)
        M_delta, p_delta = tr.TransToRp(delta_vr_transform)
        new_robot_transform = tr.RpToTrans(M_delta.dot(M_rob), p_rob + p_delta)

        if self.action_space == '3trans1rot':
            new_robot_transform = self.zero_out_pitchroll(new_robot_transform)
        if self.action_space == '3trans':
            new_robot_transform = self.zero_out_yawpitchroll(new_robot_transform)
        publish_transform(new_robot_transform, 'des_robot_transform')

        prev_target_pos, _ = self.env.get_target_state()
        delta_robot_transform = new_robot_transform.dot(tr.TransInv(prev_target_pos))
        publish_transform(delta_robot_transform, 'delta_robot_transform')
        self.prev_commanded_transform = new_robot_transform

        des_gripper_position = (1 - trigger_continuous)
        actions = tr.transform2action_local(delta_robot_transform, des_gripper_position, self.env._controller.get_cartesian_pose()[:3])

        if self.env._hp.action_mode == '3trans1rot':
            actions = np.concatenate([actions[:3], np.array([actions[5]]), np.array([des_gripper_position])])  # only use the yaw rotation
        if self.env._hp.action_mode == '3trans':
            actions = np.concatenate([actions[:3], np.array([des_gripper_position])])  # only use the yaw rotation

        if np.linalg.norm(actions[:3]) > 0.2:
            print('delta transform too large!')
            print('Press c and enter to continue')
            import pdb; pdb.set_trace()
            raise Environment_Exception

        output = {'actions': actions, 'new_robot_transform':new_robot_transform, 'delta_robot_transform': delta_robot_transform, 'policy_type': 'VRTeleopPolicy'}

        if self._hp.policy_T and self.internal_counter >= self._hp.policy_T:
            output['done'] = True

        return output

    def act(self, t=None, i_tr=None, images=None):
        return self.act_use_fixed_reference(t, i_tr, images)

    def get_default_action(self, t, i_tr, images):
        return self.get_zero_action()

    def get_zero_action(self):
        if self.env._hp.action_mode == '3trans3rot':
            actions = np.concatenate([np.zeros(6), np.array([1])])
        elif self.env._hp.action_mode == '3trans1rot':
            actions = np.concatenate([np.zeros(4), np.array([1])])
        elif self.env._hp.action_mode == '3trans':
            actions = np.concatenate([np.zeros(3), np.array([1])])
        else:
            raise NotImplementedError
        return {'actions': actions, 'new_robot_transform':np.eye(4), 'delta_robot_transform': np.eye(4), 'policy_type': 'VRTeleopPolicy'}

    def zero_out_pitchroll(self, new_robot_transform):
        rot, xyz = tr.TransToRp(new_robot_transform)
        euler = tr.rotationMatrixToEulerAngles(rot.dot(self.env._controller.default_rot.transpose()), check_error_thresh=1e-5)
        euler[:2] = np.zeros(2)  # zeroing out pitch roll
        new_rot = tr.eulerAnglesToRotationMatrix(euler).dot(self.env._controller.default_rot)
        new_robot_transform = tr.RpToTrans(new_rot, xyz)
        return new_robot_transform

    def zero_out_yawpitchroll(self, new_robot_transform):
        rot, xyz = tr.TransToRp(new_robot_transform)
        euler = tr.rotationMatrixToEulerAngles(rot.dot(self.env._controller.default_rot.transpose()), check_error_thresh=1e-5)
        euler = np.zeros(3)  # zeroing out yaw pitch roll
        new_rot = tr.eulerAnglesToRotationMatrix(euler).dot(self.env._controller.default_rot)
        new_robot_transform = tr.RpToTrans(new_rot, xyz)
        return new_robot_transform

    def oculus_to_robot(self, current_vr_transform):
        current_vr_transform = tr.RpToTrans(Quaternion(axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix,
                                            np.zeros(3)).dot(
            tr.RpToTrans(Quaternion(axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix, np.zeros(3))).dot(
            current_vr_transform)
        return current_vr_transform

    def reset(self):
        self.internal_counter = 0
        self.internal_counter_default_policy = 0
        self.prev_vr_transform = None  # used for self.act_use_deltas only

        # used for act_use_fixed_reference only:
        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None

class VRTeleopPolicyDAgger(VRTeleopPolicy):
    def __init__(self, ag_params, policyparams):
        super(VRTeleopPolicyDAgger, self).__init__(ag_params, policyparams)
        self._hp.task_ids = policyparams['model_override_params']['test_time_task_id']
        self.default_policy = self._hp.default_policy_type(ag_params, policyparams)
        self.task_idx_string = {value: key for key, value in task_string_idx.items()}

    def set_log_dir(self, dir):
        self.traj_log_dir = dir
        self.default_policy.set_log_dir(dir)

    def _default_hparams(self):
        dict = AttrDict(
            default_policy_type=None,
            policy_T=50,
            default_policy_T=50,
            task_ids=None,
        )
        default_dict = super(VRTeleopPolicyDAgger, self)._default_hparams()
        default_dict.update(dict)
        return default_dict
    
    def reset(self):
        self.random_sampled_task_id = random.choice(self._hp.task_ids)
        self.policy_desc = self.task_idx_string[self.random_sampled_task_id]
        print("Sampled task ", self.policy_desc[12:])
        self.default_policy.task_id = self.random_sampled_task_id
        return super().reset()

    def get_default_action(self, t, i_tr, images):
        output = self.default_policy.act(t=t, i_tr=i_tr, images=images)
        if self._hp.default_policy_T and self.internal_counter_default_policy >= self._hp.default_policy_T:
            output['done'] = True
        return output
    
    def act(self, t=None, i_tr=None, images=None):
        print('Human counter ', self.internal_counter)
        print('Robot counter ', self.internal_counter_default_policy)
        print('Sampled task ', self.policy_desc[12:])
        output = super(VRTeleopPolicyDAgger, self).act(t, i_tr, images)
        output['task_id'] = self.random_sampled_task_id
        output['policy_desc'] = self.policy_desc
        return output


task_string_idx = {
    "human_demo, pick up box cutter and put into drawer": 0,
    "human_demo, put fork from basket to tray": 1,
    "human_demo, put lid on stove": 2,
    "human_demo, flip pot upright which is in sink": 3,
    "human_demo, take sushi out of pan": 4,
    "human_demo, put eggplant into pot or pan": 5,
    "human_demo, put cup into pot or pan": 6,
    "human_demo, put potato in pot or pan": 7,
    "human_demo, take lid off pot or pan": 8,
    "human_demo, put big spoon from basket to tray": 9,
    "human_demo, put carrot in pot or pan": 10,
    "human_demo, put corn in pan which is on stove": 11,
    "human_demo, pick up violet Allen key": 12,
    "human_demo, take can out of pan": 13,
    "human_demo, put spatula in pan": 14,
    "human_demo, put brush into pot or pan": 15,
    "human_demo, put spoon in pot": 16,
    "human_demo, put pear in bowl": 17,
    "human_demo, put knife in pot or pan": 18,
    "human_demo, put detergent from sink into drying rack": 19,
    "human_demo, put can in pot": 20,
    "human_demo, put carrot on cutting board": 21,
    "human_demo, put banana on plate": 22,
    "human_demo, lift bowl": 23,
    "human_demo, twist knob start vertical _clockwise90": 24,
    "human_demo, put lid on pot or pan": 25,
    "human_demo, pick up red screwdriver": 26,
    "human_demo, put pot or pan in sink": 27,
    "human_demo, pick up pot from sink": 28,
    "human_demo, put pepper in pot or pan": 29,
    "human_demo, pick up pan from stove": 30,
    "human_demo, put strawberry in pot": 31,
    "human_demo, put broccoli in bowl": 32,
    "human_demo, put potato on plate": 33,
    "human_demo, put small spoon from basket to tray": 34,
    "human_demo, put pepper in pan": 35,
    "human_demo, put eggplant on plate": 36,
    "human_demo, pick up scissors and put into drawer": 37,
    "human_demo, put pot or pan from sink into drying rack": 38,
    "human_demo, put corn into bowl": 39,
    "human_demo, flip cup upright": 40,
    "human_demo, pick up glue and put into drawer": 41,
    "human_demo, pick up closest rainbow Allen key set": 42,
    "human_demo, put sweet potato in pot": 43,
    "human_demo, put spoon into pan": 44,
    "human_demo, take carrot off plate": 45,
    "human_demo, flip orange pot upright in sink": 46,
    "human_demo, take broccoli out of pan": 47,
    "human_demo, flip salt upright": 48,
    "human_demo, put sushi on plate": 49,
    "human_demo, put cup from anywhere into sink": 50,
    "human_demo, turn faucet front to left (in the eyes of the robot)": 51,
    "human_demo, put green squash into pot or pan": 52,
    "human_demo, put broccoli in pot or pan": 53,
    "human_demo, put sweet_potato in pan which is on stove": 54,
    "human_demo, put carrot on plate": 55,
    "human_demo, turn lever vertical to front": 56,
    "human_demo, put corn in pot which is in sink": 57,
    "human_demo, pick up bit holder": 58,
    "human_demo, pick up blue pen and put into drawer": 59,
    "human_demo, put knife on cutting board": 60,
    "human_demo, put red bottle in sink": 61,
    "human_demo, put pot or pan on stove": 62,
    "human_demo, put corn on plate": 63,
    "human_demo, put lemon on plate": 64,
    "human_demo, put detergent in sink": 65
}
