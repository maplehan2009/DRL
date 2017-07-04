from tensorflow.python.summary import event_accumulator
import numpy as np
import pandas as pd

if __name__ == '__main__':
	inpath0 = '/home/jingtao/Work/DRL_Data/Data_Basic_A3C/events.train0.basic'
	inpath1 = '/home/jingtao/Work/DRL_Data/Data_Basic_A3C/events.train1.basic'
	sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
		event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1}
    
	ea0 = event_accumulator.EventAccumulator(inpath0, size_guidance=sg)
	ea1 = event_accumulator.EventAccumulator(inpath1, size_guidance=sg)
	ea0.Reload()
	ea1.Reload()
	scalar_tags = ea0.Tags()['scalars']
	df0 = pd.DataFrame(columns=scalar_tags)
	df1 = pd.DataFrame(columns=scalar_tags)
    
	for tag in scalar_tags:
		events0 = ea0.Scalars(tag)
		scalars0 = np.array(list(map(lambda x: x.value, events0)))
		df0.loc[:, tag] = scalars0
		events1 = ea1.Scalars(tag)
		scalars1 = np.array(list(map(lambda x: x.value, events1)))
		df1.loc[:, tag] = scalars1
    
	df0.to_csv('/home/jingtao/Work/DRL_Data/Data_Basic_A3C/agent0_basic_a3c.csv')
	df1.to_csv('/home/jingtao/Work/DRL_Data/Data_Basic_A3C/agent1_basic_a3c.csv')
	
