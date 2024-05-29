import argparse
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

parser = argparse.ArgumentParser(description='ECG Contrastive Learning')
parser.add_argument('--pretrained', default=None, type=str, metavar="PATH", help='Path to pretrained model')
parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='Batch size for training')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='Number of epochs to train')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='Learning rate')
parser.add_argument('--norm_ecgs', default=False, type=bool, metavar='bool', help='Normalize ECGs')
parser.add_argument('--lead_groupings', default=False, type=bool, metavar='bool', help='Use lead groupings')
parser.add_argument('--seed', default=42, type=int, metavar='SEED', help='Seed for reproducibility')
parser.add_argument('--temperature', default=0.1, type=float, metavar='T', help='Temperature for contrastive loss')
parser.add_argument('--checkpoint_freq', default=10, type=int, metavar='N', help='Frequency of saving checkpoints')
parser.add_argument('--warmup_epochs', default=50, type=int, metavar='N', help='Number of warmup epochs')



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():

    args = parser.parse_args()

    seed = args.seed
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpuIds = range(torch.cuda.device_count())
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    
    batch_size = args.batch_size
    lr = args.lr
    normEcgs = args.norm_ecgs
    lead_groupings=args.lead_groupings
    currEpoch = None

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
    dataset = DataTools.PreTrainECGDatasetLoaderV2(baseDir=dataDir,patients=pre_train_patients.tolist(), augs=augs, normalize=normEcgs)
    if lead_groupings:
        assert dataset.__class__.__name__ == "PreTrainECGDatasetLoader"
    print(f"Number of ECGs in dataset: {len(dataset)}")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True)
    print(f"DataLoader creation time: {time.time() - start} seconds")

    model = Networks.BaselineConvNet(lead_grouping=lead_groupings)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if args.pretrained is not None:
        if os.path.exists(args.pretrained):
            print(f"Loading pretrained model from {args.pretrained}")
            checkpoint = torch.load(args.pretrained)
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith("module."):
                    state_dict[k[len("module."):]] = state_dict[k]
                del state_dict[k]
            model.load_state_dict(state_dict, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            currEpoch = checkpoint['epoch']

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            print(f"Pretrained model not found at {args.pretrained}. Training from scratch...")
            return

    model = torch.nn.DataParallel(model, device_ids=gpuIds)
    print(model)
    model.to(device)
    print(f"Model moved to device: {model.module.conv1.weight.device.type}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    print(f"Ready To SIMCLR Training...")
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, currEpoch=currEpoch, args=args)
    simclr.train(train_loader)

if __name__ == '__main__':
    main()
    




    





