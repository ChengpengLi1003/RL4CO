import os
import sys
import glob
import gzip
import json
import ecole
import queue
import pickle
import shutil
import argparse
import threading
import numpy as np
from pathlib import Path

from policies.exploreandstrong import ExploreThenStrongBranch


def send_orders(orders_queue, instances, seed, query_expert_prob, scip_parameters, out_dir, stop_flag):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : queue.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    rng = np.random.RandomState(seed)
    instances.seed(seed)
    episode = 0
    while not stop_flag.is_set():
        instance = next(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, scip_parameters, out_dir])
        episode += 1


def make_samples(in_queue, out_queue, stop_flag):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    sample_counter = 0
    while not stop_flag.is_set():
        episode, instance, seed, query_expert_prob, scip_parameters, out_dir = in_queue.get()

        observation_function = { 'scores': ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                 'node_observation': ecole.observation.NodeBipartite() }
        reward_function={
                "isdone": ecole.reward.IsDone(),
                "LPiteration": - ecole.reward.LpIterations(),
                "NNodes": - ecole.reward.NNodes(),
                "SolvingTime": -ecole.reward.SolvingTime(),
                "Primal": -ecole.reward.PrimalIntegral(),
                "Dual": -ecole.reward.DualIntegral(),
                "PrimalandDual":-ecole.reward.PrimalDualIntegral(),
            }
        env = ecole.environment.Branching(
            observation_function=observation_function,
            reward_function=reward_function,
            scip_params=scip_parameters
        )

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'seed': seed,
        })

        env.seed(seed)
        observation, action_set, reward_offset, done, _ = env.reset(instance)
        while not done:
            scores, scores_are_expert = observation["scores"]
            node_observation = observation["node_observation"]

            action = action_set[scores[action_set].argmax()]
            data = [node_observation,action,action_set,reward_offset,scores]

            try:
                observation, action_set, reward_offset, done, _ = env.step(action)
                if not done:
                    next_node_observation = observation['node_observation']
                    next_action_set = action_set
                else:
                    break
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving with seed {seed}\n")
                    f.write(f"{e}\n")

            if scores_are_expert and not stop_flag.is_set():
                data += [next_node_observation,next_action_set,done]
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'seed': seed,
                    'filename': filename,
                })
                sample_counter += 1

        print(f"[w {threading.current_thread().name}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_samples, n_jobs, query_expert_prob, scip_parameters):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.
    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  scip_parameters, tmp_samples_dir, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {threading.current_thread().name}] {i} / {n_samples} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # # stop all workers
    workers_stop_flag.set()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    # parameters
    node_record_prob = 0.05 # probability of running the expert strategy and collecting samples.
    time_limit = 3600 # time limit for solving each instance
    train_size = 100000 # number of samples of each type
    valid_size = 20000

    # get instances
    if args.problem == 'setcover':
        instances_train=instances_valid=ecole.instance.SetCoverGenerator()
        out_dir = 'data/samples/setcover'

    elif args.problem == 'cauctions':
        instances_train=instances_valid=ecole.instance.CombinatorialAuctionGenerator()
        out_dir = 'data/samples/cauctions'

    elif args.problem == 'indset':
        instances_train=instances_valid=ecole.instance.CapacitatedFacilityLocationGenerator()
        out_dir = 'data/samples/indset'

    elif args.problem == 'facilities':
        instances_train=instances_valid=ecole.instance.IndependentSetGenerator()
        out_dir = 'data/samples/facilities'
        node_record_prob = 0.25
        time_limit = 600

    else:
        raise NotImplementedError

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir)
    scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": time_limit,
    }
    # generate train samples
    rng = np.random.RandomState(args.seed+100)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    scip_parameters=scip_parameters)

    # generate validation samples
    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    scip_parameters=scip_parameters)
