import torch
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
import time
import os

import Loader
import DataTools
import Networks
from Simclr import SimCLR


def seed_everything(seed=42):
    # Seed Python's built-in random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For CPU operations (reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():

    args = dict(
        pretrained="runs/May22_23-48-37_cibcgpu4/checkpoint_lead_groupings_0170.pth.tar",
    )

    seed = 42
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpuIds = range(torch.cuda.device_count())
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    
    batch_size = 512
    lr = 1e-3
    normEcgs = False

    lead_groupings=True

    # Dataset
    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)
    print(f"Number of pre-train patients: {len(pre_train_patients)}")
    augmentation = [
        Loader.SpatialTransform(),
        Loader.ZeroMask(),
    ]
    augs = Loader.TwoCropsTransform(transforms.Compose(augmentation))


    start = time.time()
    print("Creating Datset...")
    dataset = DataTools.PreTrainECGDatasetLoader(baseDir=dataDir,patients=pre_train_patients.tolist(), augs=augs, normalize=normEcgs)
    print(f"Number of ECGs in dataset: {len(dataset)}")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)
    print(f"DataLoader creation time: {time.time() - start} seconds")

    
    epoch = None
    model = Networks.BaselineConvNet(lead_grouping=lead_groupings)
    optimizer = torch.optim.Adam(model.parameters(), lr)


    if args['pretrained'] is not None:
        if os.path.exists(args['pretrained']):
            print(f"Loading pretrained model from {args['pretrained']}")
            checkpoint = torch.load(args['pretrained'])
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith("module."):
                    state_dict[k[len("module."):]] = state_dict[k]
                del state_dict[k]
            model.load_state_dict(state_dict, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            print(f"Pretrained model not found at {args['pretrained']}. Training from scratch...")
            return

    epoch = 170
    model = torch.nn.DataParallel(model, device_ids=gpuIds)
    print(model)
    model.to(device)
    print(f"Model moved to device: {model.module.conv1.weight.device.type}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    print(f"Ready To SIMCLR Training...")
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, lr=lr, batch_size=batch_size, lead_groupings=lead_groupings, pretrained=args['pretrained'], epoch=epoch)
    simclr.train(train_loader)

if __name__ == '__main__':
    main()
    




    





