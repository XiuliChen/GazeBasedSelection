
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback



from envs.gaze import Gaze
from envs.utils import calc_dis
from numpy import genfromtxt

import glob
from PIL import Image

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results2(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'last average (window=100)= {np.round(y[-1],3)}')
      
timesteps = 4e6
save_feq_n=timesteps/10
fitts_W = 0.1
fitts_D=0.5 
ocular_std=0.1 
swapping_std=0.1
# Create log dir
log_dir = f'./logs/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/'
os.makedirs(log_dir, exist_ok=True)

params=np.array((fitts_D,fitts_W,ocular_std,swapping_std))
np.savetxt( f'{log_dir}params.csv', params, delimiter=',') 

# Instantiate the env
env = Gaze(fitts_W = fitts_W, 
    fitts_D=fitts_D, 
    ocular_std=ocular_std, 
    swapping_std=swapping_std)

env = Monitor(env, log_dir)

# Train the agent
model = PPO('MlpPolicy', env, verbose=1)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=save_feq_n, save_path=f'{log_dir}savedmodel/',
                                         name_prefix='eh_ppo_model')

# Train the agent

model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)

plot_results2(log_dir)
plt.savefig(f'{log_dir2}learning_curve{run}.png')
plt.close('all') 

print('Done training!!!!')

