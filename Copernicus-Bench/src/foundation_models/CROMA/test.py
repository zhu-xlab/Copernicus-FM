import torch
import pdb

from use_croma import PretrainedCROMA

model = PretrainedCROMA(
    pretrained_path="../checkpoints/CROMA_base.pt",
    size="base",
    modality="optical",
    image_resolution=120,
)
del model.GAP_FFN_s2

model = model.cuda()

inp = torch.randn([1, 12, 120, 120]).cuda()
o = model(optical_images=inp)
pdb.set_trace()
print(o.keys())
