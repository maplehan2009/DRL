import gym
import numpy as np
import matplotlib.pyplot as plt

class my_agent:
	def __init__(self):
		self.W_old = np.random.rand(2,5)
		self.W = np.random.rand(2,5)
		self.gamma = 0.9
		self.alpha = 0.01
		
	def update(self, s, s2, reward, action, done):
		if done:
			target = reward
		else:
			target = reward + self.gamma * np.max(s2.dot(self.W_old.T))
		self.W[action] = (target - s.dot(self.W[action])) * 2 * self.alpha * s + self.W[action]
		
	def update_nn(self):
		self.W_old = self.W
	
	def inference(self, observation):
		y0 = observation.dot(self.W[0])
		y1 = observation.dot(self.W[1])
		if y0 >= y1:
			return 0
		else:
			return 1

def reform_obs(observation):
	return np.append(observation, [1])

def running_mean(x):
	y = []
	n = x.size
	window = 10
	N = n - window + 1
	for i in range(N):
		y.append(np.mean(x[i:i+window]))
	return np.array(y)
		
if __name__=='__main__':
	env = gym.make('CartPole-v0')
	robot = my_agent()
	count = 0
	Reward = []
	N = 10000
	for i_episode in range(N):
		Reward_local = []
		observation = env.reset()
		s = reform_obs(observation)
		for _ in range(100):
			count += 1
			env.render()
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


