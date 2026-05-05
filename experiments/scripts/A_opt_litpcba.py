from argparse import ArgumentParser
from pathlib import Path

import mlflow.client

from rxnflow.config import Config, init_empty
from rxnflow.tasks.drug_benchmark_moo import BenchmarkTrainer
from rxnflow.utils import mlflow as mlflow_utils
from rxnflow.utils.misc import create_logger

TARGET_DIR = Path("./data/experiments/LIT-PCBA")
TARGETS = [
    "ADRB2",
    "ALDH1",
    "ESR_ago",
    "ESR_antago",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "TP53",
    "VDR",
]


if __name__ == "__main__":
    parser = ArgumentParser("RxnFlow", description="Vina-QED Optimization with GPU-accelerated UniDock")
    parser.add_argument("target", type=str, choices=TARGETS, help="LIT-PCBA target name")
    parser.add_argument("--seed", type=int, required=True, help="LIT PCBA seed")
    parser.add_argument("--mlflow-run-name", type=str, help="MLflow run name (omit to disable MLflow logging)")
    parser.add_argument(
        "--mlflow-experiment", type=str, default="rxnflow_benchmark", help="MLflow experiment name"
    )
    args = parser.parse_args()

    config = init_empty(Config())
    config.env_dir = "./data/envs/enamine_catalog/"
    config.log_dir = f"./logs/benchmark_moo/{args.target}/seed-{args.seed}"
    config.seed = args.seed
    config.print_every = 1

    config.task.docking.protein_path = TARGET_DIR / args.target / "protein.pdb"
    config.task.docking.ref_ligand_path = TARGET_DIR / args.target / "ref_ligand.mol2"
    config.task.docking.size = (22.5, 22.5, 22.5)

    # experiment setting
    config.num_training_steps = 1000
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64]
    config.algo.action_subsampling.sampling_ratio = 0.02
    config.replay.use = False

    logger = create_logger()  # non-propagate version

    if args.mlflow_run_name is not None:
        client = mlflow.client.MlflowClient(tracking_uri=mlflow_utils.MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(args.mlflow_experiment)
        exp_id = exp.experiment_id if exp is not None else client.create_experiment(args.mlflow_experiment)
        with mlflow_utils.start_mlflow_run(
            args.mlflow_run_name, group="moo", exp_id=exp_id, client=client
        ) as run:
            trainer = BenchmarkTrainer(config, mlflow_client=client, mlflow_run_id=run.info.run_id)
            trainer.run(logger)
    else:
        trainer = BenchmarkTrainer(config)
        trainer.run()
