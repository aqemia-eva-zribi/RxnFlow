from argparse import ArgumentParser

import mlflow.client

from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxyTrainer_MultiPocket
from rxnflow.utils import mlflow as mlflow_utils

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mlflow-run-name", type=str, help="MLflow run name (omit to disable MLflow logging)")
    parser.add_argument(
        "--mlflow-experiment", type=str, default=mlflow_utils.EXPERIMENT_NAME, help="MLflow experiment name"
    )
    args = parser.parse_args()

    config = init_empty(Config())
    config.env_dir = "./data/envs/catalog"
    config.task.pocket_conditional.pocket_db = "./data/experiments/CrossDocked2020/train_db.pt"
    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "CrossDocked2020")

    config.log_dir = "./logs/pocket_conditional_qvina_crossdocked2020"
    config.print_every = 10
    config.checkpoint_every = 1_000
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 8
    config.overwrite_existing_exp = True

    # change these parameters to manage memory-consumption
    config.algo.num_from_policy = 64
    config.algo.action_subsampling.sampling_ratio = 0.02

    if args.mlflow_run_name is not None:
        client = mlflow.client.MlflowClient(tracking_uri=mlflow_utils.MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(args.mlflow_experiment)
        exp_id = exp.experiment_id if exp is not None else client.create_experiment(args.mlflow_experiment)
        with mlflow_utils.start_mlflow_run(
            args.mlflow_run_name, group="pocket-conditional", exp_id=exp_id, client=client
        ) as run:
            trainer = ProxyTrainer_MultiPocket(config, mlflow_client=client, mlflow_run_id=run.info.run_id)
            trainer.run()
    else:
        trainer = ProxyTrainer_MultiPocket(config)
        trainer.run()
