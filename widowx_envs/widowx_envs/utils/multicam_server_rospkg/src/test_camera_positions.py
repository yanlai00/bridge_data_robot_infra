import rospy
import cv2
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.camera_recorder import CameraRecorder
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

if __name__ == '__main__':
    rospy.init_node("camera_rec_test")
    imtopic = IMTopic('/cam1/image_raw')
    rec = CameraRecorder(imtopic)
    rospy.sleep(1)
    # reference_image = '/mount/harddrive/spt/trainingdata/realworld/can_pushing_line/2020-09-04_09-28-29/raw/traj_group0/traj2/images{}/im_0.png'.format(cam_id)
    # cam_id = 4
    # reference_image = '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/2stage_teleop/raw/2020-11-19_15-54-42/raw/traj_group0/traj0/images{}/im_0.png'.format(cam_id)
    reference_image = '/mount/harddrive/trainingdata/robonetv2/vr_record_applied_actions_robonetv2/lift_brown_mouse/2021-04-06_16-26-22/raw/traj_group0/traj0/images0/im_0.png'
    reference_image = cv2.imread(reference_image)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    while not rospy.is_shutdown():
        tstamp, im = rec.get_image()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        ax1.imshow(im)
        ax2.imshow(reference_image)
        ax3.imshow(((im.astype(np.float32)*3 + reference_image.astype(np.float32))/4).astype(np.uint8))
        # plt.show()
        plt.pause(0.01)
        # scipy.misc.imsave('/home/sudeep/goal_images/goal_image.jpg', im)