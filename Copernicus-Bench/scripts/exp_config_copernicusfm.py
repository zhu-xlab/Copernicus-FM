import subprocess

# Define all the experiments
########################-CopernicusFM-########################
experiments = [
  
    {
        "model": "copernicusfm_seg",
        "dataset": "cobench_cloud_s2",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },         
    {
        "model": "copernicusfm_seg",
        "dataset": "cobench_cloud_s3",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.0001,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_eurosat_s1",
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },   
    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_eurosat_s2",
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },   
    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_bigearthnet_s1",
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_bigearthnet_s2",
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_lc100cls_s3",
        "task": "classification",
        "batch_size": 64,
        "lr": 1,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_seg",
        "dataset": "cobench_dfc2020_s1",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_seg",
        "dataset": "cobench_dfc2020_s2",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_seg",
        "dataset": "cobench_lc100seg_s3",
        "task": "segmentation",
        "batch_size": 16,
        "lr": 0.0001,
        "epochs": 50,
    },

    {
        "model": "copernicusfm_cd",
        "dataset": "cobench_flood_s1",
        "task": "changedetection",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 50,
    },  

    {
        "model": "copernicusfm_cls",
        "dataset": "cobench_lcz_s2",
        "task": "classification",
        "batch_size": 64,
        "lr": 0.1,
        "epochs": 50,
    },  

    {
        "model": "copernicusfm_reg",
        "dataset": "cobench_biomass_s3",
        "task": "regression",
        "batch_size": 16,
        "lr": 0.0001,
        "epochs": 50,
    },

    {
        "model": "copernicusfm_reg",
        "dataset": "cobench_aqno2_s5p",
        "task": "regression",
        "batch_size": 16,
        "lr": 0.0001,
        "epochs": 50,
    },
    {
        "model": "copernicusfm_reg",
        "dataset": "cobench_aqo3_s5p",
        "task": "regression",
        "batch_size": 16,
        "lr": 0.0001,
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
