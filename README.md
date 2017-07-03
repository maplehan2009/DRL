# DRL
Hi Thomas, 

There are two folders:

A3C is based on the project of universe-starter-agent. I want to modify its LSTM part and Loss function part in order to serve our research interests. 

Toy contains the code written by myself according to some tutorials of gym and tensorflow. RL_single shows the mechanism of gym and universe. cnn_single shows the usage of tensorflow. Now I am going to combine the two to have a DRL agent who can interact with the universe environment. You can see the details in my_cnn_agent and RL_framework. 

The link to our DRL study note: 

https://docs.google.com/document/d/1HvJXpkVhMxbL19Fx22VSWpN0L5mxlVwrIon8qvOgVQo/edit?usp=sharing

The link to my report: 

https://www.overleaf.com/read/rqsmszppcryz

Command to run the A3C agent:

`python train.py --num-workers 2 --env-id PongDeterministic-v3 --log-dir /tmp/pong`

Best,

Jingtao
