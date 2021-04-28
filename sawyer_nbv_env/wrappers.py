"""
This file implements a wrapper for merging observations of images and vectors.
"""

import numpy as np
from gym import spaces
from robosuite.wrappers import Wrapper


class MergeObsWrapper(Wrapper):
    """
    Initializes the Observation Merger wrapper. Mi
    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to robot-state and object-state.
    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """
    def __init__(self, env, img_keys=None, vect_keys=None):
        # Run super method
        super().__init__(env=env)    
        self.img_keys = img_keys
        self.vect_keys = vect_keys
        print("Merging images {} with vectors {}".format(self.img_keys,self.vect_keys))
        #self.spec = None

    def _add_merged_obs(self, ob_dict):
        images = []
        for img_key in self.img_keys:
            if len(ob_dict[img_key].shape) == 3:
                images.append(ob_dict[img_key].swapaxes(0,2).swapaxes(1,2))
            elif len(ob_dict[img_key].shape) == 2:
                images.append(np.expand_dims(ob_dict[img_key],0))
            else:
                raise ValueError("Wrong number of dimensions in the image {}".format(img_key))

        images = np.concatenate(images)

        vectors = []
        for vect_key in self.vect_keys:
            if len(ob_dict[vect_key].shape) > 1:
                raise ValueError("Expected vector observation in {}".format(vect_key))

            vectors.append(ob_dict[vect_key])
        vectors = np.concatenate(vectors)
        vectors = np.expand_dims(vectors,-1) # shape (N,1)
        vectors = np.expand_dims(vectors,-1) # shape (N,1,1)
        vectors = np.tile(vectors,(1,self.env.camera_heights[0],self.env.camera_widths[0])) # TODO select camera instead of index #0
        ob_dict["merged_obs"] = np.concatenate([images,vectors],0)
        return ob_dict


    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._add_merged_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function to add a merged observation.
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (dict) observations including merged observation
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)

        return self._add_merged_obs(ob_dict), reward, done, info        





class GymWrapper(Wrapper):
    """
    Based on the robosuite gym wrapper. Modified to return a 3D observation
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to robot-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["object-state"]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_robot-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        #self.env.spec = None # removed to catch the orignal value
        self.metadata = None

        # set up observation and action spaces
        obs = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = len(obs.shape)
        high = 255 # * np.ones(obs.shape)
        low = -255
        self.observation_space = spaces.Box(low=low, high=high, shape=obs.shape)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)


    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        if len(ob_lst) == 1:
            return ob_lst[0]
        else:
            return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def render(self, mode=None, **kwargs):
        """
        Extends env render method to catch mode selection.
        
        Args:
            mode: [NOT USED]
            **kwargs: Description
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.render()



    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
