import torch
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
import time

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

    seed = 42
    seed_everything(seed)

    normEcgs = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpuIds = range(torch.cuda.device_count())
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    
    batch_size = 512
    lr = 1e-3

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

    model = Networks.BaselineConvNet(classification=False, avg_embeddings=False)

    model = torch.nn.DataParallel(model, device_ids=gpuIds)
    print(model)
    model.to(device)
    print(f"Model moved to device: {model.module.conv1.weight.device.type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    print(f"Ready To SIMCLR Training...")
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, lr=lr, batch_size=batch_size)
    simclr.train(train_loader)

if __name__ == '__main__':
    main()
    




    





