from tensorflow.python.summary import event_accumulator
import numpy as np
import pandas as pd

if __name__ == '__main__':
	inpath = 'events.out.tfevents.1499025712.jingtao-ThinkPad-T410'
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
          event_accumulator.IMAGES: 1,
          event_accumulator.AUDIO: 1,
          event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 1}
    ea = event_accumulator.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    df = pd.DataFrame(columns=scalar_tags)
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalars = np.array(list(map(lambda x: x.value, events)))
        df.loc[:, tag] = scalars
    df.to_csv('data_from_tensorboard.csv')
    
