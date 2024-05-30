import argparse
import random
import logging

import numpy as np
import torch
import Networks
import Training
from data_utils import dataprepLVEF
import os

from torch.utils.tensorboard import SummaryWriter

import json
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = dict(
    pretrained="runs/May24_16-51-42_cibcgpu3/checkpoint_lead_groupings_0100.pth.tar",
    batch_size=512,
)

parser = argparse.ArgumentParser(description='Classification for LVEF or KCL Tasks')
parser.add_argument('--pretrained', default="runs/May27_23-23-43_cibcgpu3/checkpoint_0200.pth.tar", type=str, metavar="PATH", help='Path to pretrained model')
parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='Batch size for training')
parser.add_argument('--epochs', default=[200, 180, 50, 30, 30], type=int, nargs="+", metavar='N', help='Number of epochs to train')
parser.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs="+", metavar='N', help='Seeds for reproducibility')
parser.add_argument('--finetuning_ratios', default=[0.01, 0.05, 0.1, 0.5, 1.0], type=float, nargs="+", metavar='N', help='Finetuning Ratios')
parser.add_argument('--task', default="LVEF", type=str, metavar='N', help='Task to train on')


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
    torch.backends.cudnn.benchmark = False


def create_model(args, baseline, finetune=False):
    print("=> Creating Model")
    model = Networks.BaselineConvNet(classification=True, avg_embeddings=True)

    if baseline:
        print(f"Returning Baseline Model")
        return model

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith("module.") and not k.startswith("module.finalLayer."):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        missing_keys = {'finalLayer.0.weight', 'finalLayer.0.bias'}
        assert set(msg.missing_keys) == missing_keys
        print(f"There are {len(msg.missing_keys)} missing keys")
        
        for name, param in model.named_parameters():
            if not name.startswith("finalLayer"):
                param.requires_grad = finetune
        
        print(f"Pre-Trained Model Loaded from {args.pretrained}")
        
    else:
        print("No Pretrained Model Found")
    
    return model


def main():
    args = parser.parse_args()
    seeds =  args.seeds
    epochs = args.epochs

    results = {seed: [] for seed in seeds}

    results_file = f"results_{args.pretrained.split('/')[1]}_ep_{args.pretrained.split('/')[2].split('.')[0][-4:]}"

    writer = SummaryWriter(log_dir=f"classification_runs/{results_file}")
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'classification.log'), level=logging.DEBUG)
    print(f"Logging has been saved at {writer.log_dir}.")
    logging.info(f"Pretraining with the file {args.pretrained}")

    for seed in seeds:
        print(f"Running with seed {seed}")
        logging.info(f"Running with seed {seed}")
        seed_everything(seed)

        train_loaders, val_loader = dataprepLVEF(args)

        
        for i, train_loader in enumerate(train_loaders):
            output = {len(train_loader.dataset): {
                "Baseline": 0,
                "PreTrained/Frozen": 0,
                "PreTrained/Finetuned": 0
            }}

            for x in [0,1,2]:
                print(f"Training on {len(train_loader.dataset)} ECGs and validation on {len(val_loader.dataset)} ECGs.")
                if x == 0:
                    model = create_model(args, baseline=True)
                    lr = 1e-3
                    key = f"Baseline"
                    print(f"Training Baseline Model")
                elif x == 1:
                    model = create_model(args, baseline=False, finetune=False)
                    lr = 0.2
                    key = f"PreTrained/Frozen"
                    print(f"Training Pretrained Model with Frozen Parameters")
                elif x == 2:
                    model = create_model(args, baseline=False, finetune=True)
                    fast_lr = 0.2
                    slow_lr = 5e-5
                    key = f"PreTrained/Finetuned"
                    print(f"Training Pretrained Model with Finetuning")
                
                print(f"Requires Grad = {model.conv1.weight.requires_grad}")
                model.to(device)
                numEpoch = epochs[i]

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                if x == 2:
                    params = [{'params':getattr(model,i).parameters(), 'lr': slow_lr} if i.find("finalLayer")==-1 else {'params':getattr(model,i).parameters(), 'lr': fast_lr} for i,x in model.named_children()]
                    optimizer = torch.optim.Adam(params)

                best_auc_test = Training.train(
                    model=model,
                    trainDataLoader=train_loader,
                    testDataLoader=val_loader,
                    numEpoch=numEpoch,
                    optimizer=optimizer,
                )

                logging.info(f'For seed {seed}, with {len(train_loader.dataset)} ECGs, {key} model  BEST_AUC: {best_auc_test}')

                output[len(train_loader.dataset)][key] = best_auc_test

            
            results[seed].append(output)
            
            
    print(results)
    with open(f'results/{results_file}.json', 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    now = time.time()
    main()
    print(f"Total Time: {time.time() - now} seconds")
    
