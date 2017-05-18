import gym
import numpy as np
import matplotlib.pyplot as plt
from my_cnn_agent import my_cnn_agent
		
if __name__=='__main__':
	env = gym.make('CartPole-v0')
	robot = my_cnn_agent()
	
	count = 0
	Reward = []
	N = 10000
	for i_episode in range(N):
		Reward_local = []
		observation = env.reset()
		s = reform_obs(observation)
		for _ in range(100):
			count += 1
			#env.render()
			action = robot.inference(s)
			observation, reward, done, info = env.step(action)
			if done:
				reward = -5
			Reward_local.append(reward)
			s2 = reform_obs(observation)
			robot.update(s, s2, reward, action, done)
			if count % 10 == 0:
				robot.update_nn()
			if done:
				Reward.append(np.sum(Reward_local))
				break
	Reward = np.array(Reward) 
	y = running_mean(Reward)
	plt.scatter(range(y.size), y)
	plt.show()
