
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback



from envs.gaze import Gaze
from envs.utils import calc_dis, moving_average,plot_results2
from numpy import genfromtxt



###########################################################################
#TRAINING
###########################################################################
def main():
    # Instantiate the env
    env = Gaze(fitts_W = fitts_W,fitts_D=fitts_D, ocular_std=ocular_std, swapping_std=swapping_std)
    env = Monitor(env, log_dir)

    # Train the agent
    model = PPO('MlpPolicy', env, verbose=0, clip_range=0.15)

    '''
    # Save a checkpoint periodically
    save_feq_n=timesteps/10    
    checkpoint_callback = CheckpointCallback(save_freq=save_feq_n, save_path=f'{log_dir}savedmodel/',
        name_prefix='model_ppo')
    '''

    # Train the agent
    model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)
    
    # Save the model
    model.save(f'{log_dir}savedmodel/model_ppo')

    # Plot the learning curve
    plot_results2(log_dir)

    save_learned_behaviour()


###########################################################################
# Record Behaviour of the trained policy
###########################################################################
# save the step behaviour
# Test the trained agent

def save_learned_behaviour():
    '''
    run the trained model, return the behaviours on each step
    '''
    save_file_dir=f'{log_dir}saved_behaviours.csv'
    eps=0
    row=1

    with open(save_file_dir, mode='w') as csv_file:
        while eps<n_eps:
            obs=env.reset()
            done=False

            # save the initial obs
            saved_data=env._save_data()           
            saved_data['eps']=eps+1
 
            if row==1:
                fieldnames = saved_data.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(saved_data)

            while not done:
                action, _ = model.predict(obs,deterministic = True)
                obs, reward, done, info = env.step(action)
                saved_data=env._save_data()
                saved_data['eps']=eps+1
                writer.writerow(saved_data)
                row+=1

                if done:
                    eps+=1


            
if __name__=='__main__':


    # the target distance and width from Zhang2010 and Schutez2019
    # In the model, the start position is [0,0]
    # the display is x=[-1,1], and y=[-1,1]
    # 25.78 degree is converted to 0.5 
    # and other distance and width is converted proportionally.

    #d_zhang=np.array([25.78,11.68,6.16])

    d_zhang=np.array([11.68,6.16])
    w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

    w_schuetz=np.array([1,1.5,2,3,4,5])
    d_schuetz=np.array([5])

    unit=0.5/11.68

    w_zhang=np.round(w_zhang*unit,2)
    d_zhang=np.round(d_zhang*unit,2)

    w_schuetz=np.round(w_schuetz*unit,2)
    d_schuetz=np.round(d_schuetz*unit,2)

    ocular_std=0.08
    swapping_std=0.09

    timesteps = 2e6

    for paper in ['zhang']:
        if paper=='schuetz':
            w=w_schuetz
            d=d_schuetz
        else:
            w=w_zhang
            d=d_zhang   

        for fitts_W in w:
            for fitts_D in d:
                log_dir = f'./logs/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/'
                os.makedirs(log_dir, exist_ok=True)
                main()
