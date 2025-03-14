import subprocess

# Define all the experiments
########################-SoftCon-########################
experiments = [

    ## softcon benchmark (vit-b/14)
    {
        "model": "softcon_seg_s2",
        "dataset": "softcon/cobench_cloud_s2",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },
    {
        "model": "softcon_cls_s1",
        "dataset": "softcon/cobench_eurosat_s1", # 120
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },
    {
        "model": "softcon_cls_s2",
        "dataset": "softcon/cobench_eurosat_s2", # 120
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },
    {
        "model": "softcon_cls_s1",
        "dataset": "softcon/cobench_bigearthnet_s1", # 120
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },
    {
        "model": "softcon_cls_s2",
        "dataset": "softcon/cobench_bigearthnet_s2", # 120
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },

    {
        "model": "softcon_seg_s1",
        "dataset": "softcon/cobench_dfc2020_s1", # 256
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },
    {
        "model": "softcon_seg_s2",
        "dataset": "softcon/cobench_dfc2020_s2", # 256
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },

    {
        "model": "softcon_cd_s1",
        "dataset": "softcon/cobench_flood_s1", # 120
        "task": "changedetection",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },

    {
        "model": "softcon_cls_s2",
        "dataset": "softcon/cobench_lcz_s2", # 120
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },
]

# Run each experiment
for exp in experiments:
    print(f"Running experiment: {exp['model']} on {exp['dataset']}")
    # exp["epochs"] = 1  # This is for debug
    if "warmup_epochs" not in exp.keys():
        exp["warmup_epochs"] = 0
    subprocess.run(
        [
            "bash",
            "scripts/run.sh",  # Path to the template script
            exp["model"],
            exp["dataset"],
            exp["task"],
            str(exp["batch_size"]),
            str(exp["lr"]),
            str(exp["epochs"]),
            str(exp["warmup_epochs"]),
        ],
        check=True,
    )
    print(f"Completed: {exp['model']} on {exp['dataset']}")
