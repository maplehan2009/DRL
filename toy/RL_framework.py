import gym
import numpy as np
import matplotlib.pyplot as plt
from my_cnn_agent import my_cnn_agent
		
if __name__=='__main__':
	#make a new game environment, get the deminsion of the action space and state space
	env = gym.make('MsPacman-v0')
	n_action = env.action_space.n
	shape_state = env.observation_space.shape
	
	#initialise the agent
	agent = my_cnn_agent(n_action, shape_state)
	
	count = 0
	Reward = []
	N = 10000
	for i_episode in range(N):
		Reward_local = []
		state_prior= env.reset()
		for _ in range(100):
			count += 1
			env.render()
			action = agent.inference(state_prior)
			state_post, reward, done, info = env.step(action)
			if done:
				reward = -5
			Reward_local.append(reward)
			agent.update(state_prior, state_post, reward, action, done)
			if count % 10 == 0:
				agent.update_nn()
			if done:
				Reward.append(np.sum(Reward_local))
				break
	Reward = np.array(Reward)
