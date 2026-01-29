from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
	"""回放缓冲区，用于存储和采样经验。"""
	def __init__(self, capacity):
		"""初始化回放缓冲区，capacity 为最大容量。"""
		self.buffer = collections.deque(maxlen=capacity) 

	def add(self, state, action, reward, next_state, done): 
		"""向缓冲区添加一次转换 (s, a, r, s', done)。"""
		self.buffer.append((state, action, reward, next_state, done)) 

	def sample(self, batch_size): 
		"""从缓冲区随机采样 batch_size 个转换并返回 (states, actions, rewards, next_states, dones)。"""
		transitions = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = zip(*transitions)
		return np.array(state), action, reward, np.array(next_state), done 

	def size(self): 
		"""返回当前缓冲区中样本数量。"""
		return len(self.buffer)

def moving_average(a, window_size):
	"""计算一维数组 a 的滑动平均，处理序列两端的边界情况并返回新数组。"""
	cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
	middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
	r = np.arange(1, window_size-1, 2)
	begin = np.cumsum(a[:window_size-1])[::2] / r
	end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1] if False else (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
	return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
	"""使用 on-policy 方式训练 agent，返回每集的总回报列表。"""
	return_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
			for i_episode in range(int(num_episodes/10)):
				episode_return = 0
				transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
				state = env.reset()
				done = False
				while not done:
					action = agent.take_action(state)
					next_state, reward, done, _ = env.step(action)
					transition_dict['states'].append(state)
					transition_dict['actions'].append(action)
					transition_dict['next_states'].append(next_state)
					transition_dict['rewards'].append(reward)
					transition_dict['dones'].append(done)
					state = next_state
					episode_return += reward
				return_list.append(episode_return)
				agent.update(transition_dict)
				if (i_episode+1) % 10 == 0:
					pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
				pbar.update(1)
	return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
	"""使用 off-policy 方式训练 agent，依赖 replay_buffer，返回每集的总回报列表。"""
	return_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
			for i_episode in range(int(num_episodes/10)):
				episode_return = 0
				state = env.reset()
				done = False
				while not done:
					action = agent.take_action(state)
					next_state, reward, done, _ = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)
					state = next_state
					episode_return += reward
					if replay_buffer.size() > minimal_size:
						b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
						transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
						agent.update(transition_dict)
				return_list.append(episode_return)
				if (i_episode+1) % 10 == 0:
					pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
				pbar.update(1)
	return return_list


def compute_advantage(gamma, lmbda, td_delta):
	"""基于 td_delta（张量）计算广义优势估计（GAE），返回 torch.float 的优势张量。"""
	td_delta = td_delta.detach().numpy()
	advantage_list = []
	advantage = 0.0
	for delta in td_delta[::-1]:
		advantage = gamma * lmbda * advantage + delta
		advantage_list.append(advantage)
	advantage_list.reverse()
	return torch.tensor(advantage_list, dtype=torch.float)
