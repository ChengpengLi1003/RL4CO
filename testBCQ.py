import ecole 
from policies.gcn import GNNPolicy
from policies.BCQecole import BCQecole
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats.mstats import gmean
import argparse
import csv
def testBCQ(policy,instances,seed = 0, eval_ins_num = 50,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }
    # env = ecole.environment.Branching(
    #         observation_function=ecole.observation.NodeBipartite(),
    #         reward_function=-1.5 * ecole.reward.NNodes() ** 2,
    #         information_function={
    #             "nb_nodes": ecole.reward.NNodes(),
    #             "time": ecole.reward.SolvingTime(),
    #         },
    #         scip_params=scip_parameters,
    #         )
    env = ecole.environment.Branching(
            reward_function=
            {
                "isdone": ecole.reward.IsDone(),
                "LPiteration": - ecole.reward.LpIterations(),
                "NNodes": - ecole.reward.NNodes(),
                "SolvingTime": -ecole.reward.SolvingTime(),
                "Primal": -ecole.reward.PrimalIntegral(),
                "Dual": -ecole.reward.DualIntegral(),
                "PrimalandDual":-ecole.reward.PrimalDualIntegral(),
            },
            observation_function=ecole.observation.NodeBipartite(),
            information_function={
                "nb_nodes": ecole.reward.NNodes().cumsum(),
                "time": ecole.reward.SolvingTime().cumsum(),
                "Primal": -ecole.reward.PrimalIntegral().cumsum(),
                "Dual": -ecole.reward.DualIntegral().cumsum(),
                "PrimalandDual":-ecole.reward.PrimalDualIntegral().cumsum(),
            },
            scip_params=scip_parameters,
            )
    env.seed(seed)

    results={'nb_nodes':[],'time':[],'return':[],"PrimalandDual":[]}
    instances.seed(seed)
    print("start testing BCQ ......")
    for _, instance in zip(tqdm(range(eval_ins_num)), instances):
        gcn_return  = 0
        observation, action_set, reward_list, done, info = env.reset(instance)
        reward = reward_list["SolvingTime"]
        gcn_return += reward
        while not done:
            with torch.no_grad():
                logits = policy.select_action(torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                              torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                              torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                                              torch.from_numpy(observation.column_features.astype(np.float32)).to(device))
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, reward_list, done, info = env.step(action)
                reward = reward_list["SolvingTime"]
                gcn_return += reward
        results['nb_nodes'].append(info["nb_nodes"])
        results['time'].append(info["time"])
        results['return'].append(gcn_return)
        results['PrimalandDual'].append(info["PrimalandDual"])
    return results    

# not ready
def testGCN(behavior_policy,instances,results_dir,flag=None,seed=0,ins_num=5):
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }
    env = ecole.environment.Branching(
            observation_function=ecole.observation.NodeBipartite(),
            information_function={
                "nb_nodes": ecole.reward.NNodes().cumsum(),
                "time": ecole.reward.SolvingTime().cumsum(),
            },
            scip_params=scip_parameters,
            )
    env.seed(seed)
    instances.seed(seed)
    instance_num = 0
    result={'gcn:logits':[],'gcn:action_set':[]}
    for _, instance in zip(range(ins_num), instances):
        instance_num+=1
        # print(f'instance{instance_num}')

        observation, action_set, _, done, info = env.reset(instance)
        while not done:
            with torch.no_grad():
                observation = (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
            torch.from_numpy(observation.column_features.astype(np.float32)).to(device),
        )
                logits = behavior_policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, reward_gcn, done, info = env.step(action)
            result["gcn:logits"].append(logits)
            result['gcn:action_set'].append(action_set)
    pd.DataFrame(result).to_csv(os.path.join(results_dir,f'result_{flag}.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML4CO_MIRA-zhwang_code')
    #defualt need checking
    parser.add_argument('--num', default=10)
    args = parser.parse_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    import datetime; import os

    behaviour_policy = GNNPolicy().to(device)
    behaviour_policy.load_state_dict(torch.load("/home/yfkuang/ML4CO/offlineco/202112091017best_params.pkl"))
    
    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
 #   testGCN(behaviour_policy, instances,results_dir,flag="action_logits")

    parameters = {
        # Exploration
        "start_timesteps": 2e4,
        "initial_eps": 1,
        "end_eps": 1e-2,
        "eps_decay_period": 25e4,
        # Evaluation
        "eval_freq": 5e4,
        "eval_eps": 1e-3,
        # Learning
        "BCQ_threshold":0.05,
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.0000625,
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_frequency": 8e3,
        "tau": 1
    }
    
    bcq_path="/home/yfkuang/ML4CO/offlineco/models/threshold005_freq10_lpter_square_12-29-00_44_56"
    q_num = args.num
    # q_num = 10
    BCQ = BCQecole(
            device=device,
            behavior_policy=behaviour_policy,
            BCQ_threshold=parameters['BCQ_threshold'],
            discount = parameters['discount'],
            optimizer = parameters['optimizer'],
            optimizer_parameters = parameters['optimizer_parameters'],
            polyak_target_update = parameters['polyak_target_update'],
            target_update_frequency = parameters['target_update_frequency'],
            tau = parameters['tau'],
            initial_eps = parameters['initial_eps'],
            end_eps = parameters['end_eps'],
            eps_decay_period = parameters['eps_decay_period'],
            eval_eps = parameters['eval_eps'],
            q_path=os.path.join(bcq_path,f"q_{q_num}.pkl")
        )
    results = testBCQ(BCQ, instances,device=device)
    flag=f"q_{q_num}_thre005"
    # time = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    results_dir = os.path.join(bcq_path,"test_bcq")
    os.makedirs(results_dir,exist_ok=True)
    res = pd.DataFrame(results)
    res_path = os.path.join(results_dir,f'result_{flag}.csv')
    res.to_csv(res_path)
    mean = list(gmean(np.absolute(res),axis=0))
    mean.insert(0,"gmean")
    with open(res_path,'a+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list(mean))