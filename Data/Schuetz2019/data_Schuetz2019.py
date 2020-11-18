from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import os
import random

colors=['#d7191c',
'#fdae61',
'#2b83ba']

third_up=np.array([270,248,242,199])
third=np.array([236,211,211,174])
third_bo=np.array([202,174,180,148])


second_up=np.array([103,98,103,106,121,115,97])
second   =np.array([90,93,96,98,104,102,82])
second_bo=np.array([77,87,86,92,88,90,57])


first_up=np.array([66,56,55,55,56,56,56])
first   =np.array([53,53,52,52,53,53,53])
first_bo=np.array([47,50,49,49,50,50,50])

time_up=np.array([171,137,107,82,75,60,62])
time   =np.array([165,130,102,78,70,56,60])
time_bo=np.array([159,123,97,74,65,56,58])

size=np.array([0.5, 1,1.5,2,3,4,5])

'''
third_up=np.array([248,242,199])
third=np.array([211,211,174])
third_bo=np.array([174,180,148])


second_up=np.array([98,103,106,121,115,97])
second   =np.array([93,96,98,104,102,82])
second_bo=np.array([87,86,92,88,90,57])


first_up=np.array([56,55,55,56,56,56])
first   =np.array([53,52,52,53,53,53])
first_bo=np.array([50,49,49,50,50,50])

time_up=np.array([137,107,82,75,60,62])
time   =np.array([130,102,78,70,56,60])
time_bo=np.array([123,97,74,65,56,58])

size=np.array([1,1.5,2,3,4,5])
'''

unit=232 # 1000ms

third_time=(third/232)*1000
third_time_up=((third_up-third)/232)*1000
third_time_bo=((third-third_bo)/232)*1000

second_time=(second/232)*1000
second_time_up=((second_up-second)/232)*1000
second_time_bo=((second-second_bo)/232)*1000

first_time=(first/232)*1000
first_time_up=((first_up-first)/232)*1000
first_time_bo=((first-first_bo)/232)*1000


selection_time=(time/232)*1000
selection_time_up=((time_up-time)/232)*1000
selection_time_bo=((time-time_bo)/232)*1000



plt.errorbar(size, first_time, yerr=first_time_up,fmt='d:', color=colors[0],
             elinewidth=1, capsize=2,label='1 saccades')

plt.errorbar(size, second_time, yerr=second_time_up,fmt='s:', color=colors[1],
             elinewidth=1, capsize=2,label='2 saccades')


plt.errorbar(size[0:4], third_time, yerr=third_time_up,fmt='>:', color=colors[2],
             elinewidth=1, capsize=2,label='3+ saccades')


plt.errorbar(size, selection_time, yerr=selection_time_up,fmt='o-', color='black',
             elinewidth=1, capsize=2,label='selection time')


plt.title('Schuetz2019_Fig3')
plt.xlabel(f'target size ({chr(176)})')
plt.legend(title='Trials completed with:', loc='upper right')
plt.ylim([100 ,1200])
plt.ylabel('time (ms)')
plt.show()
plt.savefig('figures/fig3_reporduced.png')


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

ww=np.array([10,15,20,30,40,50])
plt.figure()
# plot data
size=np.array([1,1.5,2,3,4,5])

n=np.array([182,171,128,48,31,20])
up=np.array([226,205,162,70,44,30])
bo=np.array([136,137,93,27,18,9])

num_fix=1+n/349
err1=(up-n)/349
err2=(n-bo)/349

h_data=plt.errorbar(size, num_fix, yerr=err1,fmt='o-', color='black',
             ecolor='black', elinewidth=1, capsize=2,label='Data (D=5 and D=10 pooled)')

#plot model
# ocu=0.05,swp=0.05
#d5=np.array([1.37,1.1,1.0,1.0,1.0,1.0])
#d5_std=np.array([0.48,0.29,0.09,0.03,0.04,0.0])

#d10=np.array([1.80,1.50,1.25,1.06,1.0,1.0])
#d10_std=np.array([0.40,0.40,0.43,0.24,0.1,0.0])

# ocu=0.1,swp=0.1
d5=np.array([1.81,1.5,1.21,1.05,1.0,1.0])
d5_std=np.array([0.45,0.5,0.41,0.23,0.1,0.0])

d10=np.array([2.1,1.89,1.69,1.45,1.26,1.13])
d10_std=np.array([0.47,0.43,0.48 ,0.5,0.44,0.33])


h_model5=plt.plot(size, d5,'ko:', color='r',label='model d5')

h_model10=plt.plot(size, d10,'k>:', color='g',label='model d10')

plt.xlabel('Target size ')
plt.ylabel('Saccades per trial')
plt.ylim([0.9,2.5])

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0,3]
plt.legend()
plt.title('Schuetz2019_Fig4a')
plt.show()
plt.savefig('figures/fig4a_reporduced.png')

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


plt.figure()

def ID(W,D):
	return np.log2(2*D/W)

ids=[]
for dd in np.array([5,10]):
	for www in np.array([10,15,20,30,40,50]):
		ids.append(np.round(ID(www/10,dd),2))
		print(ids)

ids=np.array(ids)
ids=np.unique(ids)
print(ids)

unit1=330
up=np.array([32,36,40,36,88,135,177,216,264])
mid=np.array([18,27,26,24,68,101,140,183,218])

mid1=mid/unit1+1
up1=(up-mid)/unit1

print(mid1)
xxx
plt.errorbar(ids,mid1,yerr=up1,color='r',fmt='s-')


plt.xlabel('Index of difficulty (bits) ')
plt.ylabel('Saccades per trial')
plt.ylim([0.9,2.4])
plt.title('Schuetz2019_Fig4b')

plt.show()
plt.savefig('figures/fig4b_reporduced.png')










