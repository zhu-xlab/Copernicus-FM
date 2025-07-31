import os
import random
import torch
import webdataset as wds

class CopernicusPretrain:
    """
    WebDataset-based Copernicus dataset for PyTorch.
    """
    def __init__(self, shards_path, batch_size=2, num_workers=4, shuffle=100, shardshuffle=True, resampled=True):

        self.shards_path = shards_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle
        self.resampled = resampled

    def has_all_modalities(self, sample):
        """Ensure the sample contains all required modalities."""
        required_keys = [
            "s1_grd.pth", "s2_toa.pth", "s3_olci.pth",
            "s5p_co.pth", "s5p_no2.pth", "s5p_o3.pth",
            "s5p_so2.pth", "dem.pth", "json"
        ]
        return all(key in sample for key in required_keys)

    def sample_one_local_patch(self, sample):
        """Randomly select one local patch for S1 and S2 modalities."""
        s1, s2 = sample["s1_grd.pth"], sample["s2_toa.pth"]
        meta_s1, meta_s2 = sample["json"]["s1_grd"], sample["json"]["s2_toa"]

        idx = random.randint(0, s1.shape[0] - 1)
        sample["s1_grd.pth"], sample["s2_toa.pth"] = s1[idx], s2[idx]
        sample["json"]["s1_grd"], sample["json"]["s2_toa"] = meta_s1[idx], meta_s2[idx]
        return sample

    def sample_one_time_stamp(self, sample):
        """Randomly select one timestamp for all time-dependent modalities."""
        for key in sample:
            if key.endswith('.pth') and key != 'dem.pth':
                idx = random.randint(0, sample[key].shape[0] - 1)
                sample[key] = sample[key][idx]
                sample["json"][key.replace('.pth', '')] = sample["json"][key.replace('.pth', '')][idx]

        sample["json"]["dem"] = sample["json"]["dem"][0]
        return sample

    def get_dataloader(self):
        """Creates a WebDataset dataloader for PyTorch."""
        dataset = (
            wds.WebDataset(self.shards_path, resampled=self.resampled, shardshuffle=self.shardshuffle, nodesplitter=wds.split_by_node) 
            # shuffle shard orders and samples within shards, split by node
            .shuffle(self.shuffle) # shuffle at batch level
            .decode()
            .select(self.has_all_modalities) # select samples with all modalities
            .map(self.sample_one_local_patch) # sample one local patch for S1 and S2
            .map(self.sample_one_time_stamp) # sample one timestamp for all modalities
            .to_tuple(
                "s1_grd.pth", "s2_toa.pth", "s3_olci.pth",
                "s5p_co.pth", "s5p_no2.pth", "s5p_o3.pth",
                "s5p_so2.pth", "dem.pth", "json"
            )
            .batched(self.batch_size, partial=False)
        )

        dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.num_workers)
        
        # # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
        # number_of_batches = args.dataset_size // (args.batch_size * args.world_size)
        # data_loader_train = data_loader_train.repeat(2).slice(number_of_batches)
        # data_loader_train = data_loader_train.with_length(number_of_batches)
        # data_loader_train = data_loader_train.with_epoch(number_of_batches)

        return dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--shards_path', nargs='+', default='data/webdataset/example-{000000..000009}.tar')
    args = parser.parse_args()

    batch_size = 2
    num_workers = 2
    shuffle = 10

    copernicus_pretrain = CopernicusPretrain(args.shards_path, batch_size, num_workers, shuffle)
    dataloader = copernicus_pretrain.get_dataloader()

    for sample in dataloader:
        sample_s1, sample_s2, sample_s3, sample_co, sample_no2, sample_o3, sample_so2, sample_dem, meta = sample
        break

