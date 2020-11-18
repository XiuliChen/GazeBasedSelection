
import os
import os.path
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

colors=['#b10026',
'#e31a1c',
'#fc4e2a',
'#fd8d3c',
'#feb24c',
'#fed976',
'#ffffb2']

# the target distance and width from Zhang2010 and Schutez2019
# In the model, the start position is [0,0]
# the display is x=[-1,1], and y=[-1,1]
# 25.78 degree is converted to 0.5 
# and other distance and width is converted proportionally.

#d_zhang=np.array([25.78,11.68,6.16])
d_zhang=np.array([11.68,6.16])
w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

w_schuetz=np.array([1,1.5,2,3,4,5])
d_schuetz=np.array([10,5])

unit=0.5/11.68

w_zhang=np.round(w_zhang*unit,2)
d_zhang=np.round(d_zhang*unit,2)

w_schuetz=np.round(w_schuetz*unit,2)
d_schuetz=np.round(d_schuetz*unit,2)


###########################################################################
for paper in ['schuetz']:
    if paper=='schuetz':
        w=w_schuetz
        d=d_schuetz
        fitts_W=w[1] #W=1.5 deg
        fitts_D=d[0] #D=10 deg
    else:
        w=w_zhang
        d=d_zhang 
        fitts_W=w[1] #W=1.73 deg
        fitts_D=d[1] #D=6.16 deg 

    param_values=np.array([0.15,0.125,0.1,0.075,0.05,0.0001])
    count=-1
    for ocular_std in np.array([0.125,0.1,0.075,0.05,0.0001]):
        saccade_mean=[]
        saccade_std=[]       
        count+=1
        for swapping_std in param_values:
            print(swapping_std)

            run=1
            log_dir = f'./SensitivityTest/saved/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/run{run}/'

            no_model=False
            if os.path.exists(f'{log_dir}savedmodel/model_ppo_3000000_steps.zip'):               
                model = PPO.load(f'{log_dir}savedmodel/model_ppo_3000000_steps')
            elif os.path.exists(f'{log_dir}savedmodel/model_ppo_2000000_steps.zip'):
                model = PPO.load(f'{log_dir}savedmodel/model_ppo_2000000_steps')
            else:
                no_model=True

            env = Gaze(fitts_W = fitts_W, 
                fitts_D=fitts_D, 
                ocular_std=ocular_std, 
                swapping_std=swapping_std)

            # Test the trained agent
            n_eps=1000
            number_of_saccades=np.ndarray(shape=(n_eps,1), dtype=np.float32)
            movement_time_all=np.ndarray(shape=(n_eps,1), dtype=np.float32)
            if not no_model:
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
            

            #np.savetxt( f'{log_dir}num_saccades.csv', number_of_saccades, delimiter=',')
            #np.savetxt( f'{log_dir}movement_time.csv', movement_time_all, delimiter=',') 
            
            saccade_mean.append(np.round(np.mean(number_of_saccades),2))
            saccade_std.append(np.round(np.std(number_of_saccades),2))
            if no_model:
                saccade_mean.append(np.NaN)
                saccade_std.append(np.NaN)

        


        plt.plot(param_values, saccade_mean, 'o:',color=colors[count], label= r' $\rho_{ocular}$' f'={ocular_std}')
        #plt.errorbar(param_values, saccade_mean, yerr=saccade_std,fmt='o-', label=f'ocular noise={ocular_std}')

        plt.xlabel(r'Visual spatial noise: $\rho_{spatial}$')
        plt.ylabel('The number of saccade per trial')
        plt.ylim(0.2,2.2)
        plt.xticks(param_values)
        plt.legend(title=f'Task:D=10 {chr(176)}, W=1.5{chr(176)} \n Ocular motor noise:')
        




    plt.savefig(f'figures/sensitivity_test.png')



