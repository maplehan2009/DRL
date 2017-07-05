import pandas as pd
import matplotlib.pylab as plt
import numpy as np
plt.style.use('ggplot')

def moving_average(x):
	x = list(x)
	N = len(x)
	window = 8
	x_new = []
	for i in range(N-window):
		x_new.append(np.mean(x[i:i+window]))
	x_new = x_new + x[N-window : N]
	return np.array(x_new)	

if __name__=='__main__':
	folders = ['Basic', '1level', '2level', '3level']
	df0 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/agent0_' + folders[0] +'_a3c.csv')
	df1 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/agent1_' + folders[0] +'_a3c.csv')
	df2 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/agent0_' + folders[1] +'_a3c.csv')
	df3 = pd.read_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/agent1_' + folders[1] +'_a3c.csv')
	
	df00 = pd.concat([df0, df1])
	df00.sort_values(['step', 'reward'], inplace=True)
	df11 = pd.concat([df2, df3])
	df11.sort_values(['step', 'reward'], inplace=True)
	
	Y0 = moving_average(df00['reward'].values)
	X0 = df00['step'].values
	Y1 = moving_average(df11['reward'].values)
	X1 = df11['step'].values
	
	plt.figure(figsize=(8, 8))
	plt.plot(X0, Y0, X1, Y1, linewidth=1.2)
	plt.xlabel('Step')
	plt.ylabel('Episode Reward')
	plt.title('Episode Reward v.s. Step')
	plt.legend([folders[0], folders[1]])
	#plt.show()
	plt.savefig('/home/jingtao/Work/DRL_Data/result.eps', format='eps', dpi=1000)
	
	
