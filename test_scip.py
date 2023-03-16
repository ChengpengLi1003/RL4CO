import sys
import datetime
from numpy.random.mtrand import sample
import ipdb
import ecole
import argparse
import numpy as np
import os
import torch
import torch_geometric
import pandas as pd
from policies.gcn import GNNPolicy
import gzip
import pickle
from pathlib import Path
import csv
from tqdm import tqdm
from scipy.stats import gmean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-s', '--seed',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=1958,
    )
    parser.add_argument(
        '-n', '--num',
        help='test instances number.',
        type=int,
        default=20,
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.problem == "setcover":
        instances = ecole.instance.SetCoverGenerator()
    elif args.problem == "cauctions":
        instances = ecole.instance.CombinatorialAuctionGenerator()
    elif args.problem == "facilities":
        instances = ecole.instance.CapacitatedFacilityLocationGenerator()
    elif args.problem == "indset":
        instances = ecole.instance.IndependentSetGenerator()
    else:
        raise NotImplementedError
    
    instances.seed(args.seed)
    scip_parameters = {
                "separating/maxrounds": 0,
                "presolving/maxrestarts": 0,
                "limits/time": 3600,
            }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nnodes": ecole.reward.NNodes().cumsum(),
            "stime": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=scip_parameters
    )
    default_env = ecole.environment.Configuring(
                information_function={
                    "nnodes": ecole.reward.NNodes().cumsum(),
                    "stime": ecole.reward.SolvingTime().cumsum(),
                },
                scip_params=scip_parameters,
            )

    env_seed = [0,1,2,3,4]
    date = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    res_path = f"./testbaseline/{args.problem}_gcn&scip_s{args.seed}_{date}.csv"
    fieldnames = [
        "index",
        "seed",
        "scip:nnodes",
        "scip:stime",
        "gcn:nnodes",
        "gcn:stime"
    ]
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(f"trained_gcn_models/{args.problem}/best_params.pkl"))
    with open(res_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm(range(args.num)):
            instance = next(instances)
            for seed in env_seed:
                default_env.seed(seed)
                env.seed(seed)
                ## test scip
                default_env.reset(instance)
                _, _, _, _, scip_info = default_env.step({})
                
                ## test gcn
                observation, action_set, reward_offset, done, _ = env.reset(next(instances))
                while not done:
                    with torch.no_grad():
                        logits = policy(torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                        torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                        torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                                        torch.from_numpy(observation.column_features.astype(np.float32)).to(device))
                        action = action_set[logits[action_set.astype(np.int64)].argmax()]
                        observation, action_set, _, done, gcn_info = env.step(action)                
                
                writer.writerow({
                        "index":i,
                        "seed":seed,
                        "scip:nnodes":scip_info['nnodes'],
                        "scip:stime":round(scip_info['stime'],3),
                        "gcn:nnodes":gcn_info['nnodes'],
                        "gcn:stime":round(gcn_info['stime'],3)
                })
                csvfile.flush()


