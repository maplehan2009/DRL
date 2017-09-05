import pandas as pd
import matplotlib.pylab as plt
import numpy as np
plt.style.use('ggplot')
 

def moving_average(x, steps):
	x = list(x)
	N = len(x)
	window = 250
	x_new = []
	for i in range(N-window):
		x_new.append(np.mean(x[i:i+window]))
	#x_new = x_new + x[N-window : N]
	return np.array(x_new), steps[:N-window]	

if __name__=='__main__':
	folders = ['3level_energy_openai', '3level_maxenergy_openai', '3level_energy_deepmind', '1level_deepmind', '1level_openai', '1level_openai_argmaxsample', '3level_energy_openai_hinputI', '3level_energy_openai_hinputII', '3level_openai', '1level_openai_slowlr',
'3level_energy_openai_fb_111', '3level_energy_openai_fb_013', '3level_energy_openai_fb_310']
	#game = 'pong'
	game = 'breakout'
	#game = 'spaceinvaders'
	#game = 'seaquest'
	#game = 'qbert'
	PATH = '/home/jingtao/Work/DRL_Data/' + game + '/'
	selected_folders = [0, 10, 11, 12]
	N_folders = len(selected_folders)
	data = []
	for i in selected_folders:
		this_path = PATH + folders[i] + '/reward_data.csv'
		this_df = pd.read_csv(this_path)
		this_Y, this_X = moving_average(this_df['reward'].values, this_df['step'].values)
		data.append(this_X)
		data.append(this_Y)

	plt.figure(figsize=(8, 8))
	plt.plot(*data, linewidth=0.8)
	plt.xlim(0, 3e7)
	#plt.ylim(0, 120)
	plt.xlabel('Step')
	plt.ylabel('Episode Reward')
	plt.title('Episode Reward v.s. Step of the game: ' + game )
	
	legend = []
	for i in selected_folders:
		legend.append(folders[i])
	plt.legend(legend)
	#plt.show()
	plt.savefig(PATH + 'result.eps', format='eps', dpi=1000)
	


