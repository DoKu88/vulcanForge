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

import os

from absl import app
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

from stable_baselines3.common.env_checker import check_env

flag_dict = dict()
flag_dict['assets_root'] = '.'
flag_dict['data_dir'] = '.'
flag_dict['disp'] = False
flag_dict['shared_memory'] = False
flag_dict['task'] = 'towers-of-hanoi'
flag_dict['mode'] = 'train'
flag_dict['n'] = 1000
flag_dict['continuous'] = False
flag_dict['steps_per_seg'] = 3


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=50,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.1,
    help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(config.worker_index * config.num_workers)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return [self.cur_pos], \
            random.random() * 2 if done else -0.1, done, {}

    def seed(self, seed=None):
        random.seed(seed)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

def main():

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : cude")
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")

    device = torch.device('cpu')
    print("Device set to : cpu")

    flag_dict['assets_root'] = './ravens/environments/assets/'
    flag_dict['disp'] = False
    flag_dict['task'] = 'block-insertion'
    flag_dict['n'] = 10
    flag_dict['use_egl'] = False

    # Initialize environment and task.
    '''
    env_cls = Environment
    env = env_cls(
        flag_dict['assets_root'],
        disp = flag_dict['disp'],
        shared_memory = flag_dict['shared_memory'],
        hz=480)
    '''
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
    #ray.init(local_mode=True, ignore_reinit_error=True) # local mode for debugging
    ray.init(object_store_memory=78643200)
    agent_cams = cameras.RealSenseD415.CONFIG
    color_tuple = [
        gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
        for config in agent_cams
    ]
    depth_tuple = [
        gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
        for config in agent_cams
    ]
    position_bounds = gym.spaces.Box(
        low=np.array([0.25, -0.5, 0.], dtype=np.float32),
        high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
        shape=(3,),
        dtype=np.float32)
    config = {
        "env": Environment,
        'env_config': flag_dict,
        'num_gpus': 0,
        'num_workers': 0,
        'num_envs_per_worker': 1,
        'log_level': "WARN",
        'observation_space': gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        }),
        'action_space': gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        }),
        'framework': 'torch',
        'train_batch_size': 1,
        'sgd_minibatch_size': 1,
        'rollout_fragment_length':1
        }
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)

    env_name = "example-v0"

    print('register env...-----------------------------------------------------')
    register_env(env_name, lambda config: Environment(config))
    #check_env(Environment(config))
    print('environment checked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print('env registered!=====================================================')
    #pdb.set_trace()
    trainer = ppo.PPOTrainer(env=env_name, config=config)

    print('trainer initalized++++++++++++++++++++++++++++++++++++++++++++++++++')

    ktr = 0
    while ktr < 3:
        print(trainer.train())
        ktr += 1


if __name__ == "__main__":
    main()
