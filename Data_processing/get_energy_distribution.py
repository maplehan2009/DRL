from tensorflow.tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from os import listdir
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_df(df, events0, events1, events2):
	E0 = np.array(list(map(lambda x: x.value, events0)))
	E1 = np.array(list(map(lambda x: x.value, events1)))
	E2 = np.array(list(map(lambda x: x.value, events2)))
	steps = np.array(list(map(lambda x: x.step, events0)))
	df.loc[:, tags[0]] = steps
	df.loc[:, tags[1]] = E0
	df.loc[:, tags[2]] = E1
	df.loc[:, tags[3]] = E2
	return

 
def plot_series_E(E, name):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  sharex=True, sharey=True)
    f.set_size_inches(8, 8)
    ax1.hist(E[0:int(N / 4)], bins=150)
    ax1.set_title('Energy Evolution of option ' + name)
    ax2.hist(E[int(N / 4) : int(N / 2)], bins=150)
    ax3.hist(E[int(N / 2) : int(N * 0.75)], bins=150)
    ax4.hist(E[int(N * 0.75) : N], bins=150)

    f.savefig(PATH + folders[my_index] +  '_energy_'+ name + '.eps', format='eps', dpi=1000)   

if __name__ == '__main__':
    # Load the energy data
    folders = ['3level_energy_openai', '3level_maxenergy_openai', '3level_energy_denny', '1level_denny', '1level_openai', '1level_openai_argmaxsample', '3level_energy_openai_hinputI', '3level_energy_openai_hinputII']

    game = 'breakout'
    #game = 'spaceinvaders'
    #game = 'qbert'
    #game = 'montezuma'
    #game = 'seaquest'
    PATH = '/home/jingtao/Work/DRL_Data/' + game + '/'
    my_index = int(sys.argv[1])
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
		event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1}
        
    tags = ['step', 'energy0', 'energy1', 'energy2']	
    # only agent0 records the energy information as agent0 is the chief.
    this_path = PATH + folders[my_index] + '/train_0/'
    filename = listdir(this_path)[0]
    this_path += filename
    ea = event_accumulator.EventAccumulator(this_path, size_guidance=sg)
    ea.Reload()
    df = pd.DataFrame(columns=tags)
    events0 = ea.Scalars('model/energy0')
    events1 = ea.Scalars('model/energy1')
    events2 = ea.Scalars('model/energy2')
    get_df(df, events0, events1, events2)
    df.to_pickle(PATH + folders[my_index] + '/energy_data')
    
    # Plot the histograms
    N = len(df)
    E0 = df['energy0'].values
    E1 = df['energy1'].values
    E2 = df['energy2'].values
    
    plot_series_E(E0, '1')
    plot_series_E(E1, '2')
    plot_series_E(E2, '3')

     


    
    

	
	
	
