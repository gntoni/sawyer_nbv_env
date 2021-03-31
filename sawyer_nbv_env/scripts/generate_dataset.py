    
import os
import gym                                                                                                       
import numpy as np
from PIL import Image
from sawyer_nbv_env import sawyer_env
from robosuite.utils.transform_utils import make_pose, pose_inv, mat2pose

iterations = 1000 # x10 images per iteration
directory = "./dataset"

env = gym.make('CaptureDataset-v0')  
cams_per_iteration = env.unwrapped.n_cameras

if not os.path.exists(directory):
    os.makedirs(directory)

labelsFile  = open(os.path.join(directory,"labels.txt"), "w") 

for itr in range(iterations):
    if not (itr%50):
        print("iteration #{}".format(itr))
    obs = env.reset()        
    cube_pos = env.unwrapped.sim.data.get_body_xpos("cube_main") 
    cube_rot = env.unwrapped.sim.data.get_body_xmat("cube_main") 
    cube_pose = make_pose(cube_pos,cube_rot) # T_w_o
    
    for cam_idx in range(cams_per_iteration):
        imageCi, depthCi = env.unwrapped.sim.render(256,256,camera_name="dataset_cam_{}".format(cam_idx), depth=True) 

        # get cam_pos
        cam_pos = env.unwrapped.sim.data.get_camera_xpos("dataset_cam_{}".format(cam_idx))
        cam_rot = env.unwrapped.sim.data.get_camera_xmat("dataset_cam_{}".format(cam_idx)) 
        cam_pose = pose_inv(make_pose(cam_pos,cam_rot)) #T_ci_w

        # object pose wrt camera i
        pose_obj = cam_pose.dot(cube_pose)
        positionLabel, orientationLabel = mat2pose(cam_pose)

        # save img and label
        im = Image.fromarray(imageCi)
        dp = Image.fromarray(depthCi)
        im.save(os.path.join(directory,"image_{}.jpg".format(cam_idx+itr*cams_per_iteration)))
        dp.save(os.path.join(directory,"depth_{}.tiff".format(cam_idx+itr*cams_per_iteration)))
        labelsFile.write("image_{}.jpg\tdepth_{}.tiff\t{}\t{}\n".format(cam_idx+itr*cams_per_iteration, cam_idx+itr*cams_per_iteration, positionLabel, orientationLabel))
labelsFile.close()

