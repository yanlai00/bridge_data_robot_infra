import numpy as np


class GraspPolicy:

    def __init__(self, env, pick_height_thresh=0.12, xyz_action_scale=10.0,
                 pick_point_noise=0.00, pick_point_z=0.029):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.pick_point_z = pick_point_z
        # self.reset()

    def reset(self, pick_point):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)

        self.pick_point = pick_point
        # self.pick_point = bullet.get_object_position(
        #     self.env.objects[self.object_to_target])[0]
        # if self.object_to_target in GRASP_OFFSETS.keys():
        #     self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point += np.random.normal(scale=self.pick_point_noise, size=(3,))
        self.pick_point[2] = self.pick_point_z
        self.grasp_executed = False

    def get_action(self):
        ee_pos = self.env._get_obs()['ee_coord']

        # object_pos, _ = bullet.get_object_position(
        #     self.env.objects[self.object_to_target])
        object_lifted = ee_pos[2] > self.pick_height_thresh
        gripper_pickpoint_xy_dist = np.linalg.norm(self.pick_point[:2] - ee_pos[:2])
        done = False
        neutral_action = [0.]

        if gripper_pickpoint_xy_dist > 0.02 and not self.grasp_executed:
            # print('moving near obj')
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [1.0]
            status = 'approaching'
        elif ee_pos[2] - self.pick_point_z > 0.001 and not self.grasp_executed:
            # print('moving near obj, down')
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [1.0]
            status = 'approaching'
        elif not self.grasp_executed:
            # print('executing grasp')
            # near the object, performs grasping action
            # action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            self.grasp_executed = True
            status = 'grasping'
        elif not object_lifted:
            # print('lifting')
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.startpos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            status = 'lifting'
        else:
            # print('holding')
            # Hold
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            status = 'holding'

        agent_info = dict(done=done, status=status)
        # action = np.concatenate(
        #     (action_xyz, action_angles, action_gripper, neutral_action))

        action = np.concatenate((action_xyz, action_gripper))

        return action, agent_info

    def drop_object(self, drop_point):

        self.grasp_executed = False

        for i in range(20):
            ee_pos = self.env._get_obs()['ee_coord']
            gripper_pickpoint_xy_dist = np.linalg.norm(drop_point[:2] - ee_pos[:2])

            if (gripper_pickpoint_xy_dist > 0.02 or ee_pos[2] - self.pick_point_z > 0.001) and not self.grasp_executed:
                # print('moving near obj')
                action_xyz = (drop_point - ee_pos) * self.xyz_action_scale
                action_gripper = [0.0]
                status = 'approaching'
            elif not self.grasp_executed:
                action_xyz = [0., 0., 0.]
                action_gripper = [1.0]
                self.grasp_executed = True
                status = 'un-grasping'
            else:
                return

            action = np.concatenate((action_xyz, action_gripper))
            o, r, done, info = self.env.step(action)
