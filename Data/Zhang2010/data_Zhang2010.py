'''
This file is to produce the Figure3A in Zhang et al., 2010
The figure shows The eye movement time and eye pointing time.
I recreated by eye-ball the 

IMPORTANT NOTE!!!
(1)They used the dwell time of 800 ms to highlight and
select targets
(2) Experiment 1: 
19-inch CRT display (1024×768 pixels)--> 1280 pixel == 19 inch (48.26cm)
The subject sat about 70 cm in front of the screen (1857).
Target diameter W are: 40, 56, 70, 86, 100 pixels.
Amplitude A: 200, 380,and 850 pixels.

Convert to visual angles:

Target diameter W are: 1.23, 1.73, 2.16, 2.65, 3.08 degrees.
Amplitude A: 6.16, 11.68,and 25.78 degrees.

'''
import matplotlib.pyplot as plt
import numpy as np


colors=['#253494',
'#2c7fb8',
'#41b6c4']

# I converted the target size and amplitude from the number of pixels
# to visual angle in degree
target_size=np.array([1.23,1.73,2.16,2.65,3.08])
target_amplitude=np.array([6.16, 11.68,25.78])

unit1=32 # 500ms
fig=plt.figure(figsize=(14,5))
ai=3
for D in target_amplitude:
	plt.subplot(1,3,ai)
	ai-=1
	
	if D==target_amplitude[0]:
		time_up=np.array([38,32,27,23,23])
		time   =np.array([36,30,25,21,22])

		dwell_time=[1075,969,918,890,876]

	elif D==target_amplitude[1]:
		time_up=np.array([39,35,29,25,25])
		time   =np.array([37,32,26,23,23])

		dwell_time=[1271,1101,943,933,930]

	elif D==target_amplitude[2]:
		time_up=np.array([49,43,36,30,28])
		time   =np.array([46,40,33,28,26])

		dwell_time=[1258,958,948,950,898]


	if D==target_amplitude[0]:
		ptime_up=np.array([109,96,87,81,79])
		ptime   =np.array([106,93,84,79,78])
	elif D==target_amplitude[1]:
		ptime_up=np.array([122,108,90,86,86])
		ptime   =np.array([119,103,88,84,84])
	elif D==target_amplitude[2]:
		ptime_up=np.array([132,105,98,93,87])
		ptime   =np.array([128,102,95,90,85])

	selection_time=(time/unit1)*500
	selection_time_up=((time_up-time)/unit1)*500

	p_time=(ptime/unit1)*500
	p_time_up=((ptime_up-ptime)/unit1)*500

	EMT=(time/unit1)*500
	EPT=(ptime/unit1)*500

	width=0.3
	ax=fig.gca()

	ax.bar(target_size, EMT, width, color='y', edgecolor='black', hatch="/",label='Eye movement time (data)')

	plt.bar(target_size, EPT-EMT, width, 
	             bottom=EMT, color=colors[1],label='Selection time (data)')

	plt.errorbar(target_size, selection_time, yerr=selection_time_up,fmt=':', color='r')
	plt.errorbar(target_size, p_time, yerr=p_time_up,fmt=':', color='r')

	plt.xlabel(f'target size ({chr(176)})')
	plt.ylabel('Time (ms)')
	plt.legend(loc='upper right')
	plt.ylim([200,2250])
	plt.yticks(np.arange(0,2251,250))
	plt.xticks(target_size)
	plt.title(f'target distance={D}{chr(176)}')

	
	for i in range(5):
		ax.text(target_size[i]-0.1,800,f'{dwell_time[i]}',rotation=90,fontsize=18)

plt.show()
plt.savefig(f'figures/Fig3a_zhang2010.png')



