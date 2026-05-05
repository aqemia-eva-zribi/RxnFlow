from argparse import ArgumentParser

import mlflow.client

from rxnflow.config import Config, init_empty
from rxnflow.tasks.seh import SEHTrainer
from rxnflow.utils import mlflow as mlflow_utils


def parse_args():
    parser = ArgumentParser("RxnFlow", description="SEH Proxy optimization")
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=10_000,
        help="Number of training iterations (default: 10,000)",
    )
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.01,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    run_cfg.add_argument("--mlflow-run-name", type=str, help="MLflow run name (omit to disable MLflow logging)")
    run_cfg.add_argument(
        "--mlflow-experiment", type=str, default=mlflow_utils.EXPERIMENT_NAME, help="MLflow experiment name"
    )
    run_cfg.add_argument("--debug", action="store_true", help="For debugging option")
    return parser.parse_args()


def run(args):
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.log_dir = args.out_dir
    config.print_every = 10
    config.num_training_steps = args.num_iterations
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 2000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 5000

    if args.debug:
        config.overwrite_existing_exp = True

    if args.mlflow_run_name is not None:
        client = mlflow.client.MlflowClient(tracking_uri=mlflow_utils.MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(args.mlflow_experiment)
        exp_id = exp.experiment_id if exp is not None else client.create_experiment(args.mlflow_experiment)
        with mlflow_utils.start_mlflow_run(
            args.mlflow_run_name, group="seh", exp_id=exp_id, client=client
        ) as run:
            trainer = SEHTrainer(config, mlflow_client=client, mlflow_run_id=run.info.run_id)
            trainer.run()
            trainer.terminate()
    else:
        trainer = SEHTrainer(config)
        trainer.run()
        trainer.terminate()


if __name__ == "__main__":
    args = parse_args()
    run(args)
