# Deep Meta Reinforcement Learning Model

Folder 'toy' contains some tutorial files. cnn_single.py introduces a simple implementation of CNN in tensorflow. RL_single introduces the usage of the reinforcement learning environment gym and universe. RL_framework and my_cnn_agent.py together demonstrate one implementation of deep reinforcement learning agent. Other two files about LSTM show the implementation of Stacked LSTM architectures.

Folder 'Data processing' contains the files to extract data from Tensorboard and the files to plot the figures based on the extracted data.

Folder 'Agent' contains the files to implement the Deep RL agent. The loss function is defined in A3C.py. The NN architecture of the agent is defined in model.py. In model.py, LSTMPolicy_alpha represents a one level LSTM + A3C agent. LSTMPolicy_beta represents a 3 levels LSTM agent with energy regularisation. LSTMPolicy_gamma represents an improved LSTMPolicy_beta agent by taking the previous h vector as the input vector of the current time step.
 
The link to the DRL study note: 

https://docs.google.com/document/d/1HvJXpkVhMxbL19Fx22VSWpN0L5mxlVwrIon8qvOgVQo/edit?usp=sharing

The link to my final report: 

https://www.overleaf.com/read/vvvqxqbbtjck

Command to run the A3C agent:

`python train.py --num-workers 4 --env-id PongDeterministic-v3 --log-dir ~/DRL/A3C/pong`
