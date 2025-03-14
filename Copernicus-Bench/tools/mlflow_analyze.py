import os
import mlflow
import pandas as pd


def collect_best_lightning_metrics(logs_dir, mode="min"):
    """
    Collect best metrics for PyTorch Lightning MLflow runs.

    Args:
        logs_dir (str): Base directory containing model run folders
        metric_name (str): Primary metric to sort runs by
        mode (str): 'min' or 'max' for sorting metric

    Returns:
        pd.DataFrame: Consolidated metrics for best runs
    """
    best_runs = []

    # Iterate through model folders
    for model_folder in os.listdir(logs_dir):
        model_path = os.path.join(logs_dir, model_folder, "mlruns")

        if not os.path.exists(model_path):
            continue

        mlflow.set_tracking_uri(model_path)

        if "_cls_" in model_path:
            metric_name = "test_acc1_epoch"
        elif "_seg_" in model_path:
            metric_name = "test_miou_epoch"
        else:
            raise ValueError("Not supported task")

        # Search all experiments
        experiments = mlflow.search_experiments()

        for exp in experiments:
            # Search runs in the experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if not runs.empty:
                # Sort runs by specified metric
                is_ascending = mode == "min"
                sorted_runs = runs.sort_values(
                    by=f"metrics.{metric_name}", ascending=is_ascending
                )

                # Get the best run
                best_run = sorted_runs.iloc[0]

                # Collect run metrics and parameters
                run_info = {
                    "model_folder": model_folder,
                    "experiment_name": exp.name,
                    "run_id": best_run["run_id"],
                    **{k: v for k, v in best_run.items() if "metrics.test" in k},
                }

                # Add hyperparameters
                collect_hyper_params = [
                    "param_params.args/lr",
                    "param_params.args/batch_size",
                    "param_params.args/model",
                    "param_params.args/dataset",
                ]
                run_info.update(
                    {
                        f"param_{k}": v.strip()
                        for k, v in best_run.items()
                        if k in collect_hyper_params
                    }
                )

                best_runs.append(run_info)

    return pd.DataFrame(best_runs)


def save_best_metrics(
    logs_dir,
    output_file="best_lightning_metrics.csv",
    metric_name="val_loss",
    mode="min",
):
    """
    Save best PyTorch Lightning metrics to CSV.

    Args:
        logs_dir (str): Base directory containing model run folders
        output_file (str): Path to save output CSV
        metric_name (str): Primary metric to sort runs by
        mode (str): 'min' or 'max' for sorting metric
    """
    best_metrics = collect_best_lightning_metrics(logs_dir, mode=mode)

    # Save to CSV and JSON
    best_metrics.to_csv(output_file, index=False)
    best_metrics.to_json(output_file.replace(".csv", ".json"), orient="records")

    print(f"Best metrics saved to {output_file}")


# Example usage
if __name__ == "__main__":
    logs_directory = "./logs"

    save_best_metrics(
        logs_directory,
        mode="max",  # Use 'min' for loss, 'max' for accuracy
    )
