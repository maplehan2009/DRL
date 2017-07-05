from tensorflow.python.summary import event_accumulator
import numpy as np
import pandas as pd

def get_df(df, events):
	rewards = np.array(list(map(lambda x: x.value, events)))
	steps = np.array(list(map(lambda x: x.step, events)))
	df.loc[:, tags[0]] = steps
	df.loc[:, tags[1]] = rewards
	return

if __name__ == '__main__':
	folders = ['Basic', '1level', '2level', '3level']
	#folder = folders[1]
	inpath0 = '/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/train_0/events.out.tfevents.1499180217.cloud-vm-45-26.doc.ic.ac.uk'
	inpath1 = '/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/train_1/events.out.tfevents.1499180253.cloud-vm-45-26.doc.ic.ac.uk'
	
	inpath2 = '/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/train_0/events.out.tfevents.1499210582.cloud-vm-47-204.doc.ic.ac.uk'
	inpath3 = '/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/train_1/events.out.tfevents.1499210587.cloud-vm-47-204.doc.ic.ac.uk'
	
	sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
		event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1}
    
	ea0 = event_accumulator.EventAccumulator(inpath0, size_guidance=sg)
	ea1 = event_accumulator.EventAccumulator(inpath1, size_guidance=sg)
	ea2 = event_accumulator.EventAccumulator(inpath2, size_guidance=sg)
	ea3 = event_accumulator.EventAccumulator(inpath3, size_guidance=sg)	
	
	ea0.Reload()
	ea1.Reload()
	ea2.Reload()
	ea3.Reload()
	
	tags = ['step', 'reward']
	df0 = pd.DataFrame(columns=tags)
	df1 = pd.DataFrame(columns=tags)
	df2 = pd.DataFrame(columns=tags)
	df3 = pd.DataFrame(columns=tags)
	
	events0 = ea0.Scalars('global/episode_reward')
	events1 = ea1.Scalars('global/episode_reward')
	events2 = ea2.Scalars('global/episode_reward')
	events3 = ea3.Scalars('global/episode_reward')
	
	get_df(df0, events0)
	get_df(df1, events1)
	get_df(df2, events2)
	get_df(df3, events3)
    
	df0.to_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/agent0_' + folders[0] +'_a3c.csv')
	df1.to_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[0] + '_A3C/agent1_' + folders[0] + '_a3c.csv')
	df2.to_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/agent0_' + folders[1] +'_a3c.csv')
	df3.to_csv('/home/jingtao/Work/DRL_Data/Data_' + folders[1] + '_A3C/agent1_' + folders[1] + '_a3c.csv')	
	
