
## Made for exp_12_revcubedetect.sh

import torch
from robosuite.utils.transform_utils import make_pose, pose_inv, mat2pose, get_orientation_error
from simple_posereg.dataloader  import data_transform


def test_reward(self):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.
        cube_present = False

        # sparse completion reward  TODO
        #if self._check_success():
        #    reward = 2.25

        # use a shaping reward
        #elif self.reward_shaping:

        ## sparse reward

        # Occlusion detection
        self.segmentation_img = self.sim.render(256,256,camera_name="robot0_eye_in_hand", segmentation=True)
        #check if the cube is present in the image
        if self.cube_vis_geom_id in self.segmentation_img[:,:,1]:
            cube_present = True

        # Label
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        cube_pos = self.sim.data.get_body_xpos("cube_main") 
        cube_rot = self.sim.data.get_body_xmat("cube_main") 
        cube_pose = make_pose(cube_pos,cube_rot) # T_w_o
        cam_pos = self.sim.data.get_camera_xpos("robot0_eye_in_hand")
        cam_rot = self.sim.data.get_camera_xmat("robot0_eye_in_hand") 
        cam_pose = pose_inv(make_pose(cam_pos,cam_rot)) #T_ci_w
        pose_obj = cam_pose.dot(cube_pose) # object pose wrt camera i
        positionLabel, orientationLabel = mat2pose(cam_pose)

        # Prediction
        image, depth = self.sim.render(256,256,camera_name="robot0_eye_in_hand", depth=True) 
        input_img = data_transform(image)
        input_img = torch.unsqueeze(input_img, 0)
        depth_img = self.to_tensor(depth)
        depth_img = torch.unsqueeze(depth_img, 0)   
        inputs =  torch.cat([input_img, depth_img], 1)
        with torch.no_grad():
            out_pose = self.pose_regressor(inputs)
            self.pose_reward = self.pose_loss(out_pose, torch.cat((torch.Tensor(positionLabel),torch.Tensor(orientationLabel))).unsqueeze(0))

        reward = -self.pose_reward.item()

        if not cube_present:
            reward -= 1.0

        return reward