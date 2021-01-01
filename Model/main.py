
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

import glob
from PIL import Image



###########################################################################
#TRAINING
###########################################################################
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

###########################################################################
for paper in ['zhang']:
    if paper=='schuetz':
        w=w_schuetz
        d=d_schuetz
    else:
        w=w_zhang
        d=d_zhang   

    for fitts_W in w:
        for fitts_D in d:


            
            for run in [1]:
                # Create log dir
                log_dir = f'./logs4/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/run{run}/'
                log_dir2 = f'./logs4/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/'

                os.makedirs(log_dir, exist_ok=True)

                params=np.array((fitts_D,fitts_W,ocular_std,swapping_std,timesteps))
                np.savetxt( f'{log_dir}params.csv', params, delimiter=',') 

                # Instantiate the env
                env = Gaze(fitts_W = fitts_W, 
                    fitts_D=fitts_D, 
                    ocular_std=ocular_std, 
                    swapping_std=swapping_std)

                env = Monitor(env, log_dir)

                # Custom MLP policy of two layers of size 32 each with tanh activation function
                #policy_kwargs = dict(net_arch=[128, 128])
                #policy_kwargs=policy_kwargs
               
                # Train the agent
                
                model = PPO('MlpPolicy', env, verbose=0, clip_range=0.15)


                save_feq_n=timesteps/10

                # Save a checkpoint every 1000 steps
                checkpoint_callback = CheckpointCallback(save_freq=save_feq_n, save_path=f'{log_dir}savedmodel/',
                                                         name_prefix='model_ppo')

                # Train the agent

                model.learn(total_timesteps=int(timesteps), callback=checkpoint_callback)
                model.save(f'{log_dir}savedmodel/model_ppo')

                plot_results2(log_dir)


                print('Done training!!!!')




                ###########################################################################
                # Record Behaviour of the trained policy
                ###########################################################################
                # save the step data

              
                print('Saving Data!!!!')
                # Test the trained agent
                n_eps=1000
                number_of_saccades=np.ndarray(shape=(n_eps,1), dtype=np.float32)
                movement_time_all=np.ndarray(shape=(n_eps,1), dtype=np.float32)
                eps=0
                while eps<n_eps:                
                        done=False
                        step=0
                        obs= env.reset()
                        fixate=np.array([0,0])
                        movement_time=0
                        while not done:
                            step+=1
                            action, _ = model.predict(obs,deterministic = True)
                            obs, reward, done, info = env.step(action)
                            move_dis=calc_dis(info['fixate'],fixate)
                            fixate=info['fixate']
                            movement_time+=37+2.7*move_dis
                            if done:
                                number_of_saccades[eps]=step
                                movement_time_all[eps]=movement_time
                                eps+=1
                                break
                

                np.savetxt( f'{log_dir}num_saccades.csv', number_of_saccades, delimiter=',')
                np.savetxt( f'{log_dir}movement_time.csv', movement_time_all, delimiter=',') 

                plt.title(f'num_sccade={np.round(np.mean(number_of_saccades),2)},std_sccade={np.round(np.std(number_of_saccades),2)} \n MT={np.round(np.mean(movement_time_all),2)}, MT_std={np.round(np.mean(movement_time_all),2)}')

                plt.savefig(f'{log_dir2}learning_curve{run}.png')
                plt.close('all') 
            



            ###########################################################################
            # Record Behaviour of the trained policy
            ###########################################################################
            # save the step data
            '''
            print('Saving Data!!!!')
            # Test the trained agent


            # for saving the data
            nrows=20000
            saved_data=np.ndarray(shape=(nrows,10), dtype=np.float32)
            n_eps=1000
            number_of_saccades=np.ndarray(shape=(n_eps,1), dtype=np.float32)

            row=0
            eps=0

            with open(f'{log_dir}steps_data_verbose.csv', mode='w') as csv_file:    
                while eps<n_eps and row<nrows-1:
                    
                    done=False
                    step=0
                    obs= env.reset()
                    while not done:
                        step+=1
                        action, _ = model.predict(obs,deterministic = True)
                        obs, reward, done, info = env.step(action)
                    

                        info['eps']=eps
                        
                        
                        row+=1

                        saved_data[row,:]=[info['n_fixation'], #0
                            info['target'][0],#1
                            info['target'][1],#2
                            info['belief'][0],#3
                            info['belief'][1],#4
                            info['aim'][0],#3
                            info['aim'][1],#4
                            info['fixate'][0],#6
                            info['fixate'][1],#7
                            eps]


                        
                        
                        fieldnames = info.keys()
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        
                        if row==1:
                            writer.writeheader()

                        writer.writerow(info)

                        if done:
                            number_of_saccades[eps]=step
                            eps+=1
                            break
                    
            saved_data=saved_data[0:row]
            np.savetxt( f'{log_dir}steps_data.csv', saved_data, delimiter=',') 
            np.savetxt( f'{log_dir}num_saccades.csv', number_of_saccades, delimiter=',') 
            '''

