import numpy as np
import torch
import json

from solvers.BaselineSolver import BaselineSolver

from solvers.VIBSolver import VIBSolver
from utils.GeneralUtils import init_logger

__CONFIG__ = 'config.json'
SEEDS = 5


def get_solver(config):
    if config['solver']['model_name'] in 'baseline':
        return BaselineSolver(config)
    elif 'VIB' in config['solver']['model_name']:
        return VIBSolver(config)
    return None


def main(config):
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    for seed in range(SEEDS):
        config['log_name'] = f"{config['solver']['model_name']}_{config['solver']['dataset_name']}_{seed}"
        init_logger(config['log_name'], config['solver']['is_testing'])
        config['seed'] = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        solver = get_solver(config)

        if config['mode'] == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()
        else:
            return 0


if __name__ == "__main__":
    with open(__CONFIG__, 'r') as f:
        config = json.load(f)
    main(config)
