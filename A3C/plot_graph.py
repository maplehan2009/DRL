import pandas as pd
import matplotlib.pylab as plt
import numpy as np
plt.style.use('ggplot')

def moving_average(x):
	N = len(x)
	window = 8
	x_new = []
	for i in range(N-window):
		x_new.append(np.mean(x[i:i+window]))
	return np.array(x_new)	

if __name__=='__main__':
	df0 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_Basic_A3C/agent0_basic_a3c.csv')
	df1 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_Basic_A3C/agent1_basic_a3c.csv')
	Y0 = df0['global/episode_reward'].values
	Y1 = df1['global/episode_reward'].values
	Y0 = moving_average(Y0)
	Y1 = moving_average(Y1)
	N0 = Y0.size
	N1 = Y1.size
	X0 = range(N0)
	X1 = range(N1)

	
	plt.figure(figsize=(8, 8))
	plt.plot(X0, Y0, X1, Y1, linewidth=1.2)
	plt.xlabel('Episode')
	plt.ylabel('Episode Reward')
	plt.title('Episode Reward v.s. Episode')
	plt.legend(['Agent0', 'Agent1'])
	#plt.show()
	plt.savefig('/home/jingtao/Work/DRL_Data/Data_Basic_A3C/reward.eps', format='eps', dpi=1000)
	
	
