import sys
sys.path.append("/misc/dl00/halo/plausibl/pacer/pacer")
import os
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import optuna
import sqlite3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--dataset", type=str, default="JTA", help="Dataset name. among JRDB, JTA")
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize', study_name="my_study", storage=f"sqlite:///./experiments/{args.dataset}/{args.exp_name}/study.db", load_if_exists=True, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    study = optuna.load_study(study_name="my_study", storage=f"sqlite:///./experiments/{args.dataset}/{args.exp_name}/study.db",)
    print("Number of finished trials: {}".format(len(study.trials)))
    print(f"Best ADE: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    fig1, fig2 = optuna.visualization.plot_slice(study), optuna.visualization.plot_optimization_history(study)
    fig1.write_image(f"experiments/{args.dataset}/{args.exp_name}/optuna_slice.png")
    fig2.write_image(f"experiments/{args.dataset}/{args.exp_name}/optuna_optimization_history.png")
    print("All done.")