"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
# https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
# For action space issue
# https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
import os

#from absl import app
#from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment_RL import ContinuousEnvironment
from ravens.environments.environment_RL import Environment
from ravens.tasks import cameras

import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import sys
import pdb
from gym.envs.registration import register
import torch

# stable_baselines3 Imports:
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env

# RLLIB Imports:
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


flag_dict = dict()
flag_dict['assets_root'] = './ravens/environments/assets/'
flag_dict['disp'] = False
flag_dict['task'] = 'block-insertion'
flag_dict['n'] = 10
flag_dict['use_egl'] = False
flag_dict['assets_root'] = '.'
flag_dict['data_dir'] = '.'
flag_dict['continuous'] = False
flag_dict['steps_per_seg'] = 3
flag_dict['mode'] = 'train'

def main():

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : cuda")
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")

    flag_dict['assets_root'] = './ravens/environments/assets/'
    flag_dict['disp'] = True
    flag_dict['task'] = 'block-insertion_1'
    flag_dict['n'] = 10
    flag_dict['use_egl'] = False

    # Initialize environment and task.
    task = tasks.names[flag_dict['task']](continuous = flag_dict['continuous'])
    task.mode = flag_dict['mode']

    # Initialize scripted oracle agent and dataset.
    #agent = task.oracle(env, steps_per_seg=flag_dict['steps_per_seg'])
    dataset = Dataset(os.path.join(flag_dict['data_dir'], flag_dict['task'] + '-' + task.mode))
    flag_dict['task'] = task

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
      seed = -1 if (task.mode == 'test') else -2

    # Determine max steps per episode.
    max_steps = task.max_steps
    if flag_dict['continuous']:
      max_steps *= (flag_dict['steps_per_seg'] * agent.num_poses)


    # above is from demos.py pretty much =======================================
    # bottom from: https://docs.ray.io/en/latest/rllib-env.html

    agent_cams = cameras.RealSenseD415.CONFIG
    #color_tuple = [
    #    gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
    #    for config in agent_cams
    #]
    #depth_tuple = [
    #    gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
    #    for config in agent_cams
    #]
    config_cam = agent_cams[0]
    color_tuple = gym.spaces.Box(0, 255, config_cam['image_size'] + (3,), dtype=np.uint8)
    depth_tuple = gym.spaces.Box(0, 255, config_cam['image_size'] + (3,), dtype=np.uint8)

    position_orientation_bounds = gym.spaces.Box(
        low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        dtype=np.float32)

    print('flag_dict')
    print(flag_dict)
    config = {
        "env": Environment,
        'env_config': flag_dict,
        'num_gpus': 0,
        'num_workers': 0,
        'num_envs_per_worker': 1,
        'log_level': "WARN",
        'observation_space': color_tuple,
        #'observation_space': gym.spaces.Dict({
        #    'color': color_tuple,
        #    'depth': depth_tuple,
        #}),
        'action_space' : position_orientation_bounds,
        'framework': 'torch',
        'train_batch_size': 1,
        'sgd_minibatch_size': 1,
        'rollout_fragment_length':1
    }

    env_name = "example-v0"

    # ==========================================================================
    # stable_baselines3 experiment =============================================
    # NOTE: supports dict observations but not nested observation spaces :(
    '''
    env = Environment(flag_dict)
    model = PPO('CnnPolicy', env)
    model.learn(total_timesteps=100) # log_interval=1, tb_log_name=folder_name, callback=callbacks
    '''

    # ==========================================================================
    # RLLIB experiment =========================================================
    #ray.init(local_mode=True, ignore_reinit_error=True) # local mode for debugging

    ray.init(local_mode=True)
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)
    register_env(env_name, lambda config: Environment(config))
    trainer = ppo.PPOTrainer(env=env_name, config=config)
    print('config space: ', trainer.get_config()['action_space'])

    ktr = 0
    while ktr < 10:
        print('training step: ', ktr)
        print(trainer.train())
        ktr += 1



if __name__ == "__main__":
    main()
