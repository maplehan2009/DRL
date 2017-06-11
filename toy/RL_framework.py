import gym
import numpy as np
import matplotlib.pyplot as plt
from my_cnn_agent import my_cnn_agent
		
BATCH_SIZE = 4

def stack_image(s):
    return np.mean(s, axis= 2)

if __name__=='__main__':
    #make a new game environment, get the deminsion of the action space and state space
    env = gym.make('MsPacman-v0')
    n_action = env.action_space.n
    shape_state = env.observation_space.shape

    #initialise the agent
    agent = my_cnn_agent(n_action, shape_state, BATCH_SIZE)

    N = 10
    print('Explore Phase')
    for i_episode in range(N):
        state_prior = env.reset()
        state_prior = stack_image(state_prior)
        for _ in range(2000):
            env.render()
            action = agent.inference(state_prior)
            state_post, reward, done, info = env.step(action)
            state_post = stack_image(state_post)
            if done:
                reward = -5
            agent.get_memory(state_prior, action, reward, state_post, done)
            if done:
                break
            else:
                state_prior = state_post
                 
    print('Learning Phase')
    N = 10
    Reward = []
    Reward_temp = 0
    for i_episode in range(N):
        print(i_episode)
        Reward.append(Reward_temp)
        Reward_temp = 0
        state_prior = env.reset()
        state_prior = stack_image(state_prior)
        for _ in range(100):
            #env.render()
            action = agent.inference(state_prior)
            state_post, reward, done, info = env.step(action)
            state_post = stack_image(state_post)
            if done:
                reward = -5
            Reward_temp += reward
            agent.get_memory(state_prior, action, reward, state_post, done)
            agent.train_my_agent()
            if done:
                break
            else:
                state_prior = state_post
    Reward = np.array(Reward)
    plt.scatter(range(N), Reward)
    plt.show()
