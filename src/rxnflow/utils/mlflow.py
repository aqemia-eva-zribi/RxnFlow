"""Utilities to log RxnFlow training runs to MLflow.

This module follows the conventions used in `aqemia.adme_multitask.training.mlflow`:
the MLflow tracking URI is exposed as a module-level `Final` constant; the
`MlflowClient` is created and passed explicitly by callers; helpers do not rely
on global state.
"""

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Final

import mlflow.client
import mlflow.entities

MLFLOW_TRACKING_URI: Final = "https://mlflow-prod.taileedd2a.ts.net/"
EXPERIMENT_NAME: Final = "rxnflow"


@contextlib.contextmanager
def start_mlflow_run(
    name: str,
    *,
    group: str | None = None,
    exp_id: str,
    client: mlflow.client.MlflowClient,
) -> Iterator[mlflow.entities.Run]:
    """Context manager to create an MLflow run and mark it FINISHED/FAILED on exit.

    Args:
        name: Run name to register in MLflow.
        group: Optional logical grouping tag, written under the `group` MLflow tag.
            This mirrors the `group` argument of `wandb.init()` from the previous
            backend, with no first-class equivalent in MLflow.
        exp_id: The experiment in which to create the new MLflow run.
        client: The MLflow client to use to communicate with the tracking server.

    Yields:
        The created MLflow run.
    """
    tags = {}
    if group is not None:
        tags["group"] = group
    run = client.create_run(experiment_id=exp_id, run_name=name, tags=tags)
    try:
        yield run
    except BaseException:
        client.set_terminated(run_id=run.info.run_id, status="FAILED")
        raise
    else:
        client.set_terminated(run_id=run.info.run_id, status="FINISHED")


def log_params(
    run_id: str,
    params: Mapping[str, Any],
    *,
    client: mlflow.client.MlflowClient,
    prefix: str = "",
) -> None:
    """Recursively flatten and log nested parameters to MLflow.

    MLflow parameters must be flat key/value pairs, so nested config structures
    such as `{"opt": {"learning_rate": 1e-4}}` are converted to dotted keys like
    `"opt.learning_rate"` before being logged. None values are skipped.

    Args:
        run_id: Target MLflow run identifier.
        params: Nested parameter mapping to flatten and log.
        client: MLflow client used to record parameters.
        prefix: Optional prefix prepended to each logged parameter key during the
            recursive traversal.
    """
    for name, value in params.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            log_params(run_id, value, client=client, prefix=key)
        elif value is None:
            continue
        else:
            client.log_param(run_id, key, value)


def log_metrics(
    run_id: str,
    metrics: Mapping[str, Any],
    *,
    step: int,
    client: mlflow.client.MlflowClient,
) -> None:
    """Log a batch of scalar metrics to MLflow at a given step.

    Non-scalar values (lists, arrays, mappings) are silently skipped, since the
    MLflow `log_metric` API only accepts floats. This matches what wandb did
    implicitly in the previous backend: per-iteration `info` dictionaries
    sometimes contain non-scalar entries that we did not log meaningfully either.

    Args:
        run_id: Target MLflow run identifier.
        metrics: Mapping from metric key to scalar value.
        step: Training step at which the metrics were computed.
        client: MLflow client used to record the metrics.
    """
    for key, value in metrics.items():
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        client.log_metric(run_id=run_id, key=key, value=float_value, step=step)
