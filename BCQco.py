import sys
import copy
# from numpy.random.mtrand import sample
import datetime
# import ipdb
# from BCQ.discrete_BCQ.main import interact_with_environment
# sys.path.append('../')
# sys.path.append('..')
import ecole
import argparse
import numpy as np
import os
import torch
import torch_geometric
import pandas as pd
# import env
import gzip
import pickle
from pathlib import Path
# import utils
from policies.BCQecole import BCQecole
from policies.exploreandstrong import ExploreThenStrongBranch
from policies.gcn import *
from data_buffer import ReplayBuffer
from testBCQ import testBCQ
import csv
import time
from torch.utils.tensorboard import SummaryWriter


def train_BCQ(env,instances,sample_loader,device,args,behavior_policy,parameters,results_dir,model_dir,tb_writer,test_baseline=True):
    ## 初始化并保存策略
    policy = BCQecole(
        device=device,
        behavior_policy=behavior_policy,
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
    )
    torch.save(policy.Q.state_dict(),os.path.join(model_dir,f"q_start.pt"))
    scip_parameters = {
            "separating/maxrounds": 0,
            "presolving/maxrestarts": 0,
            "limits/time": 3600,
        }
    ## 测试GCN和BCQ基准水平
    test_baseline = False
    if test_baseline:
        print('test baseline...')
        print('-------------------------------------------------')
        results={"gcn:nb_nodes":[],"gcn:time":[],"gcn:return":[],
                 "scip:nb_nodes":[],"scip:time":[],"scip:return":[]}
        # env for gcn
        env_gcn = ecole.environment.Branching(
            reward_function=-1.5 * ecole.reward.NNodes() ** 2,
            observation_function=ecole.observation.NodeBipartite(),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_parameters,
            )
        env_gcn.seed(args.seed)
        # env for scip
        default_env = ecole.environment.Configuring(
            observation_function=None,
            reward_function=-1.5 * ecole.reward.NNodes() ** 2,
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_parameters)
        default_env.seed(args.seed)
        instances.seed(args.seed)#测试instance和训练不一样
        for instance_count, instance in zip(range(50), instances):
            #run GCN 
            gcn_return = gcn_time = gcn_nb_nodes = 0
            observation, action_set, _, done, info = env_gcn.reset(instance)
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
                    observation, action_set, reward_gcn, done, info = env_gcn.step(action)
                    gcn_return += reward_gcn
                    gcn_nb_nodes += info["nb_nodes"]
                    gcn_time += info["time"]
            results['gcn:nb_nodes'].append(gcn_nb_nodes)
            results["gcn:return"].append(gcn_return)
            results['gcn:time'].append(gcn_time)

            # Run SCIP's default brancher
            default_env.reset(instance)
            _, _, scip_reward, _, default_info = default_env.step({})
            results['scip:nb_nodes'].append(default_info['nb_nodes'])
            results["scip:return"].append(scip_reward)
            results['scip:time'].append(default_info['time'])
        pd.DataFrame(results).to_csv(os.path.join(results_dir, f'baseline.csv'))

    # 开始训练BCQ策略
    nb_nodes_dir = os.path.join(results_dir,"BCQ_nb_nodes.csv")
    time_dir = os.path.join(results_dir,"BCQ_time.csv")
    return_dir =os.path.join(results_dir,"BCQ_return.csv")
    pdual_dir = os.path.join(results_dir,"BCQ_pdual.csv")
    iter_step = 0
    for epoch in range(args.max_epoch):
        start = time.time()
        ### 测试BCQ性能
        results = testBCQ(policy,instances,args.seed,args.eval_ins_num,device)
        with open(nb_nodes_dir,'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(results['nb_nodes'])
        with open(time_dir,'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(results['time'])
        with open(return_dir,'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(results['return'])
        with open(pdual_dir,'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(results['PrimalandDual'])
        end = time.time()
        print(f'epoch{epoch} testing consumes{(end-start)/60}min')
        print('-----------------------------------------------------')
        print(f'start training for epoch{epoch+1}')
        start = time.time()
        policy.train(sample_loader,writer=tb_writer,iter_step=iter_step)
        torch.save(policy.Q.state_dict(),os.path.join(model_dir,f"q_{epoch+1}.pt"))
        end = time.time()
        print(f'epoch{epoch+1} training consumes{(end-start)/60}min')
    tb_writer.close()

        

def interact_with_environment(env,instances,args):
    episode_counter, sample_counter = 0, 0

    while sample_counter < args.buffer_size:
        episode_counter += 1
        observation, action_set, reward_offset, done, _ = env.reset(next(instances))
        while not done:
            (scores, scores_are_expert), node_observation = observation
            action = action_set[scores[action_set].argmax()]
        # Only save samples if they are coming from the expert (strong branching)
            # if scores_are_expert and (sample_counter < args.buffer_size):
            if scores_are_expert and (sample_counter < args.buffer_size):
                sample_counter += 1
                next_observation, next_action_set, reward_offset, done, _ = env.step(action)
                if not done:
                    (next_scores, next_scores_are_expert), next_node_observation = next_observation
                else:
                    next_node_observation = node_observation
                    next_action_set = action_set
                data = [node_observation,action,action_set,scores,next_node_observation,next_action_set,reward_offset,done]
                # print(action)
                # store single sampe
                filename = f"buffer/buffer2_{args.env}/{args.buffer_name}_{args.env}_{args.seed}_{sample_counter+190000}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)
                observation = copy.copy(next_observation)
                action_set = copy.copy(next_action_set)
            else:
            # replay_buffer.add_samples()
                observation, action_set, reward_offset, done, _ = env.step(action)

            # observation, action_set, _, done, _ = env.step(action)
        print(f"Episode {episode_counter}, {sample_counter} samples collected so far")
#     #for saving files 
#     setting = setting = f"{args.env}_{args.seed}"
#     buffer_name = f"{args.buffer_name}_{setting}"
#     #Initialize and load policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ML4CO_MIRA-zhwang_code')
    #defualt need checking
    parser.add_argument('--env', default='SetCoverGenerator', help='environment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--buffer_name', default='origin', help='buffer name')
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size')
    parser.add_argument("--train_behavioral", action="store_true")  
    parser.add_argument("--generate_buffer", action="store_true")   
    parser.add_argument("--train_policy", action="store_true")
    parser.add_argument("--max_epoch", default=30, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--batch_size", default=32, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--eval_ins_num", default=5, type=int)
    parser.add_argument("--flag",type=str, default =None)
    parser.add_argument("--BCQ_threshold", default=0.2, type = float)
    parser.add_argument("--BCQ_tar_freq", default=10, type = int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--reward_type", default="PrimalandDual", type=str, choices=["isdone","LPiteration","NNodes","SolvingTime","Primal","Dual","PrimalandDual"])
    parser.add_argument("--device",default=3)
    parser.add_argument("--hist_interval",default=100)
    
    args = parser.parse_args()
    
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
        "BCQ_threshold": args.BCQ_threshold,
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr":args.lr,  #"lr": 0.0000625
			"eps": 0.00015
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_frequency": args.BCQ_tar_freq,
		"tau": 1
	}

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }

    print("---------------------------------------")	
    if args.generate_buffer:
        print("generate_buffer")
        print("---------------------------------------")
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    elif args.train_behavioral:
        print("train_behavioral")
        print("---------------------------------------")
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    assert not (args.train_behavioral and args.generate_buffer), "Train_behavioral and generate_buffer cannot both be true."
    
    # if not os.path.exists("./buffers_{args.buffer_size}"):
    #     os.makedirs("./buffers_{args.buffer_size}")
    # Path(f"buffers_{args.buffer_size}").mkdir(exist_ok=True)
    instances = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
    # Path(f"buffer/buffer1_{args.env}").mkdir(exist_ok=True)

    # %采集数据环境
    env0 = ecole.environment.Branching(
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
            observation_function=(ExploreThenStrongBranch(expert_probability=1),
                ecole.observation.NodeBipartite()),
            information_function={
                "nb_nodes": ecole.reward.NNodes().cumsum(),
                "time": ecole.reward.SolvingTime().cumsum(),
                "Primal": -ecole.reward.PrimalIntegral().cumsum(),
                "Dual": -ecole.reward.DualIntegral().cumsum(),
                "PrimalandDual":-ecole.reward.PrimalDualIntegral().cumsum(),
            },
            scip_params=scip_parameters,
            )
    # 测试环境
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

    env0.seed(args.seed)
    env.seed(args.seed)
    # default_env.seed(args.seed)
    #给instance加随机种子
    instances.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    #采集数据
    # if args.generate_buffer==True:
    if 0.5<1:
        # print("start generating buffer...")
        # interact_with_environment(env0, instances, args)
    # elif args.train_policy==True:
        behaviour_policy = GNNPolicy().to(device)
        behaviour_policy.load_state_dict(torch.load("/home/yfkuang/ML4CO/offlineco/trained_gcn_models/setcover/best_params.pkl"))
        # sample_files = [str(path) for path in Path(f"buffers_{args.buffer_size}").glob(f"{args.buffer_name}_{args.env}_{args.seed}_*.pkl")]
        # sample_files = [str(path) for path in Path("/home/chengpeng/ML4CO_MIRA-zhwang_code/ML4CO_MIRA-zhwang_code/data_generation/buffers_100000").glob("origin_SetCoverGenerator_0_*.pkl")]
        sample_files = [str(path) for path in Path("/home/yfkuang/ML4CO/offlineco/buffer/buffer1_SetCoverGenerator").glob("origin_SetCoverGenerator_0_*.pkl")]
        sample_data = GraphDataset(sample_files,args.reward_type)   
        # sample_files = sample_files[: int(0.1 * len(sample_files))]
        # ipdb.set_trace()
        sample_loader = torch_geometric.loader.DataLoader(sample_data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        date = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        results_dir = os.path.join("./results",f'{args.flag}_{date}')
        model_dir = os.path.join("./models",f'{args.flag}_{date}')
        os.makedirs(results_dir); os.makedirs(model_dir)
        tb_writer = SummaryWriter(os.path.join(results_dir,"tensorboard"))
        train_BCQ(env,instances,sample_loader,device,args,behaviour_policy,parameters,results_dir,model_dir,tb_writer=tb_writer)




        