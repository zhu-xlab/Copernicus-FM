"""Factory utily functions to create datasets and models."""

from foundation_models import (
    CromaModel,
    ScaleMAEModel,
    GFMModel,
    SoftConModel,
    DofaModel,
    SatMAEModel,
    CopernicusFMModel,
    ViTModel
)

from datasets.cobench_clouds2_wrapper import CoBenchCloudS2Dataset
from datasets.cobench_eurosats2_wrapper import CoBenchEuroSATS2Dataset
from datasets.cobench_eurosats1_wrapper import CoBenchEuroSATS1Dataset
from datasets.cobench_bigearthnets12_wrapper import CoBenchBigEarthNetS12Dataset
from datasets.cobench_lc100clss3_wrapper import CoBenchLC100ClsS3Dataset
from datasets.cobench_lc100segs3_wrapper import CoBenchLC100SegS3Dataset
from datasets.cobench_airqualitys5p_wrapper import CoBenchAirQualityS5PDataset
from datasets.cobench_clouds3_wrapper import CoBenchCloudS3Dataset
from datasets.cobench_dfc2020s12_wrapper import CoBenchDFC2020S12Dataset
from datasets.cobench_lczs12_wrapper import CoBenchLCZS12Dataset
from datasets.cobench_biomasss3_wrapper import CoBenchBiomassS3Dataset
from datasets.cobench_floods1_wrapper import CoBenchFloodS1Dataset

model_registry = {
    "croma": CromaModel,
    "scalemae": ScaleMAEModel,
    "gfm": GFMModel,
    "softcon": SoftConModel,
    "dofa": DofaModel,
    "satmae": SatMAEModel,
    "copernicusfm": CopernicusFMModel,
    "vit": ViTModel,
}

dataset_registry = {
    "cobench_cloud_s2": CoBenchCloudS2Dataset,
    "cobench_eurosat_s2": CoBenchEuroSATS2Dataset,
    "cobench_eurosat_s1": CoBenchEuroSATS1Dataset,
    "cobench_bigearthnet_s12": CoBenchBigEarthNetS12Dataset,
    "cobench_lc100cls_s3": CoBenchLC100ClsS3Dataset,
    "cobench_lc100seg_s3": CoBenchLC100SegS3Dataset,
    "cobench_airquality_s5p": CoBenchAirQualityS5PDataset,
    "cobench_cloud_s3": CoBenchCloudS3Dataset,
    "cobench_dfc2020_s12": CoBenchDFC2020S12Dataset,
    "cobench_lcz_s12": CoBenchLCZS12Dataset,
    "cobench_biomass_s3": CoBenchBiomassS3Dataset,
    "cobench_flood_s1": CoBenchFloodS1Dataset,
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    dataset = dataset_class(config_data)
    # return the train, val, and test dataset
    return dataset.create_dataset()


def create_model(args, config_model, dataset_config=None):
    model_name = config_model.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    model = model_class(args, config_model, dataset_config)

    return model
