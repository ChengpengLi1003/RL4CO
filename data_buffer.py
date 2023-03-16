import numpy as np

# observation 每个是class 对象，存储queue of class，不能用queue，因为队列出队受限，没法自由sample
# 还是得用list of items
# GraphDataset 中的sample files 替换为replay_buffer即可

#暂时没用上，使用dataloader训练，而不是buffer
class ReplayBuffer():
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self._init_key_values()
        self._cur_sample_num = 0
        
    def _init_key_values(self):
        # 为何RL 算法中数据一般都会提前分配好内存，这会很大影响计算开销吗？
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'done': []
        }

    def add_samples(self, state, action, next_state, reward, done):
        self.dataset['observations'].append(state)
        self.dataset['actions'].append(action)
        self.dataset['next_observations'].append(next_state)
        self.dataset['rewards'].append(reward)
        self.dataset['done'].append(done)
        # add samples to buffer
        # sample_len = len(samples['observations'])
        # if self._cur_sample_num + sample_len <= self.max_size:
        #     for k,v in samples.items():
        #         self.dataset[k].extend(v)
        #     self._cur_sample_num += sample_len
        # else:
        #     for k,v in samples.items():
        #         if self.max_size - self._cur_sample_num > 0:
        #             self.dataset[k].extend(v[:self.max_size-self._cur_sample_num])
        #         self.dataset[k][:sample_len-self.max_size+self._cur_sample_num] = v[self.max_size-self._cur_sample_num:] 
        #     self._cur_sample_num = self.max_size


    def random_batch(self, batch_size):
        # sample batch samples from buffer
        batch_indexes = np.random.randint(self._cur_sample_num, size=batch_size)
        batch_samples = {}
        for k,v in self.dataset.items():
            # TODO: check 是否需要shuffle
            sample = [v[batch_indexes[i]] for i in range(batch_size)]
            batch_samples[k] = sample
        
        return batch_samples