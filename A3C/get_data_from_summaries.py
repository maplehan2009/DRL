from tensorflow.tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from os import listdir

def get_df(df, events):
	rewards = np.array(list(map(lambda x: x.value, events)))
	steps = np.array(list(map(lambda x: x.step, events)))
	df.loc[:, tags[0]] = steps
	df.loc[:, tags[1]] = rewards
	return

if __name__ == '__main__':
	folders = ['Data_1level_A3C', 'Data_3level_A3C', 'Data_3level_energy_A3C', 'max_energy']
	game = 'pong'
	#game = 'pacman'
	#game = 'breakout'
	#game = 'spaceinvader'
	#game = 'qbert'
	PATH = '/home/jingtao/Work/DRL_Data/' + game + '/'
	selected_folders = [2, 3]
	N_workers = 4
	df = []
		
	sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
		event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1}
	tags = ['step', 'reward']
	for i in selected_folders:
		for j in range(N_workers):
			this_path = PATH + folders[i] + '/train_' + str(j) + '/'
			filename = listdir(this_path)[0]
			this_path += filename
			ea = event_accumulator.EventAccumulator(this_path, size_guidance=sg)
			ea.Reload()
			this_df = pd.DataFrame(columns=tags)
			events = ea.Scalars('global/episode_reward')
			get_df(this_df, events)
			df.append(this_df)
		df_total = pd.concat(df)
		df_total.sort_values(tags, inplace=True)
		df_total.to_csv(PATH + folders[i] + '/reward_data.csv')
		df = []
		del df_total
		
