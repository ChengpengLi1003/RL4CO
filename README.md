Process of using this code:
1. Collect dataset: Run the generate_dataset.py to use scip solver to collect mixed data of expert policy and random policy with ecole framework;
2. Training a behaviour policy: Run the train_gcn.py to learn a behaviour policy with imitation learning;
3. Train a Reinforcement learning policy: Run the BCQco.py to combine the behaviour policy and DQN to get a better policy. 

Note: More offline algorithm will be introduced soon.
