import argparse
import random
import logging

import numpy as np
import torch
import Networks
import Training
from data_utils import dataprepGenetics
import os

from torch.utils.tensorboard import SummaryWriter
import parameters
import json
import time
import wandb
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpuIds = range(torch.cuda.device_count())
os.environ["WANDB_API_KEY"] = "e56acefffc20a7f826010f436f392b067f4e0ae5"

parser = argparse.ArgumentParser(description='Classification for Genetic Results')
parser.add_argument('--pretrained', default=None, type=str, metavar="PATH", help='Path to pretrained model')
parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='Batch size for training')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of epochs to train')
parser.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs="+", metavar='N', help='Seeds for reproducibility')
parser.add_argument('--lr', default=[1e-4, 0.02, 1e-3], type=float, nargs="+", metavar='N', help='Learning Rate as [lr, fast_lr, slow_lr]')
parser.add_argument('--num_workers', default=32, type=int, metavar='N', help='Number of workers for data loading')
parser.add_argument('--arch', default='ECG_SpatioTemporalNet1D', choices=["ECG_SpatioTemporalNet1D", "BaselineConvNet"], type=str, metavar='ARCH', help='Architecture to use')
parser.add_argument('--logtowandb', default=False, type=bool, metavar='bool', help='Log to wandb')
parser.add_argument('--lead_groupings', action='store_true', help='Use lead groupings')
parser.add_argument('--numECGs', default="1", type=str, metavar='N', help='Number of ECGs to use')

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
    if not args.lead_groupings:
        if args.arch == "BaselineConvNet":
            model = Networks.BaselineConvNet(classification=True, avg_embeddings=True)
        elif args.arch =="ECG_SpatioTemporalNet1D":
            model = Networks.ECG_SpatioTemporalNet1D(**parameters.spatioTemporalParams_1D, classification=True, avg_embeddings=True)
    else:
        lead_groups = [
            [0,1,6,7],
            [2,3,4,5]
        ]
        model = Networks.ModelGroup(arch=args.arch, parameters=parameters.spatioTemporalParams_1D, lead_groups=lead_groups, classification=True )


    if baseline:
        print(f"Returning Baseline Model")
        return model

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']

        if not args.lead_groupings:
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
        else:
            model_g1_state_dict = state_dict['model_g1']
            model_g2_state_dict = state_dict['model_g2']

            msg = model.model_g1.load_state_dict(model_g1_state_dict, strict=False)
            if (len(msg.missing_keys) != 0): print(f"There are {len(msg.missing_keys)} missing keys")
            
            for name, param in model.model_g1.named_parameters():
                if not name.startswith("finalLayer"):
                    param.requires_grad = finetune

            msg = model.model_g2.load_state_dict(model_g2_state_dict, strict=False)
            if (len(msg.missing_keys) != 0): print(f"There are {len(msg.missing_keys)} missing keys")

            
            for name, param in model.model_g2.named_parameters():
                if not name.startswith("finalLayer"):
                    param.requires_grad = finetune
            
        print(f"Pre-Trained Model Loaded from {args.pretrained}")
        
    else:
        print("No Pretrained Model Found")
    
    return model

def main():
    args = parser.parse_args()
    seeds = args.seeds
    epoch = args.epochs

    results = {seed: [] for seed in seeds}
    lr = args.lr[0]
    # results_file = f"results_{"Genetics"}_{args.pretrained.split('/')[1]}_ep_{args.pretrained.split('/')[2].split('.')[0][-4:]}"
    results_file = f"results_{args.pretrained.split('/')[1]}_ep_{args.pretrained.split('/')[2].split('.')[0][-4:]}_ECGS_{args.numECGs}"

    writer = SummaryWriter(log_dir=f"classification_runs_genetics/{results_file}")
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'classification.log'), level=logging.INFO)
    logging.info(f"Starting Classification for Genetics with {args.pretrained} at {time.ctime()}")
    logging.info(f"args: {args}")

    for seed in seeds:
        print(f"Running with seed {seed}")
        logging.info(f"Running with seed {seed}")
        seed_everything(seed)

        train_loader, val_loader = dataprepGenetics(args, seed)

        training_size = len(train_loader.dataset)
        logging.info(f"Training on {training_size} ECGs and validation on {len(val_loader.dataset)} ECGs.")
        for x in [0,1,2]:
            print(f"Training on {training_size} ECGs and validation on {len(val_loader.dataset)} ECGs.")
            
            if x == 0:
                model = create_model(args, baseline=True)
                lr = args.lr[0]
                print("Training the baseline Model")
                key = "baseline"
            elif x == 1:
                model = create_model(args, baseline=False, finetune=False)
                lr = args.lr[1]
                key = "PreTrained-Frozen"
                print("Training the model with frozen weights")
            elif x == 2:
                model = create_model(args, baseline=False, finetune=True)
                fast_lr = args.lr[1]
                slow_lr = args.lr[2]
                key = "PreTrained-Finetuned"
                print("Training the model with Finetuning")
           

            check = model.model_g1 if args.lead_groupings else model
            print(f"Requires Grad = {check.conv1.weight.requires_grad if args.arch == 'BaselineConvNet' else check.firstLayer[0].weight.requires_grad}")
            model = torch.nn.DataParallel(model, device_ids=gpuIds)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            if x == 2:
                    params = [{'params':getattr(model,i).parameters(), 'lr': slow_lr} if i.find("finalLayer")==-1 else {'params':getattr(model,i).parameters(), 'lr': fast_lr} for i,x in model.named_children()]
                    optimizer = torch.optim.Adam(params)                              
            
            # if args.logtowandb:
            #             wandbrun = wandb.init(
            #                 project=results_file,
            #                 notes=f"Seed {seed}, with {training_size} ECGs, {key} model",
            #                 config = dict(vars(args)),
            #                 entity="deekshith",
            #                 reinit=True,
            #                 name=f"{seed}_{int(args.finetuning_ratios[i]*100)}_perc_{key}"
            #             )

            now = time.time()

            print("Training Baseline Model")
            best_auc_test, best_acc, best_acc_f1max =  Training.trainGenetics(
                model=model,
                trainDataLoader=train_loader,
                testDataLoader=val_loader,
                numEpoch=epoch,
                optimizer=optimizer,
                modelSaveDir=writer.log_dir,
                modelName=f"{seed}_{args.arch}_model_{key}",
                logToTensorBoard=True,
                logToWandB=False
            )

            logging.info(f"For seed {seed}, {key} model best AUC on test set is {best_auc_test}")
    
if __name__ == "__main__":
    main()