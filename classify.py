import random
import logging

import numpy as np
import torch
import pickle
import DataTools
import Networks
import Training
import os

from torch.utils.tensorboard import SummaryWriter

import json
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = dict(
    pretrained="runs/May24_16-51-42_cibcgpu3/checkpoint_lead_groupings_0100.pth.tar",
    batch_size=512,
)

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

def dataprep(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = True

    print("Preparing Data For Finetuning")
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)
    
    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)
    
    num_classification_patients = len(pre_train_patients)
    finetuning_ratios =  [0.01, 0.05, 0.1, 0.5, 1.0]
    num_finetuning = [int(num_classification_patients * ratio) for ratio in finetuning_ratios]
    print(f"Number of Finetuning Patients: {num_finetuning}")

    patientInds = list(range(num_classification_patients))
    random.shuffle(patientInds)

    train_loaders = []
    dataset_lengths = []
    for i in num_finetuning:
        finetuning_patient_indices = patientInds[:i]
        finetuning_patients = pre_train_patients[finetuning_patient_indices]

        dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=finetuning_patients.tolist(), normalize=normEcgs)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args["batch_size"],
            num_workers=32,
            shuffle=True,
            pin_memory=True,
        )

        train_loaders.append(loader)
        dataset_lengths.append(len(dataset))
    
    validation_dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args["batch_size"],
        num_workers=32,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Preparing Finetuning with {dataset_lengths} number of ECGs and validation with {len(validation_dataset)} number of ECGs")

    return train_loaders, val_loader

def create_model(args, baseline):
    print("=> Creating Model")
    model = Networks.BaselineConvNet(classification=True, avg_embeddings=False)

    if baseline:
        print(f"Returning Baseline Model")
        return model

    if args["pretrained"] is not None:
        checkpoint = torch.load(args["pretrained"], map_location="cpu")
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
                param.requires_grad = False
        
        print(f"Pre-Trained Model Loaded from {args['pretrained']}")
        
    else:
        print("No Pretrained Model Found")
    
    return model


def main(args):
    seeds =  [42, 43, 44, 45, 46]
    results = {seed: [] for seed in seeds}
    epochs = [250, 250, 50, 30, 30]
    results_file = f"results_{args['pretrained'].split('/')[1]}_ep_{args['pretrained'].split('/')[2].split('.')[0][-4:]}"

    writer = SummaryWriter(log_dir=f"classification_runs/{results_file}")
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'classification.log'), level=logging.DEBUG)
    print(f"Logging has been saved at {writer.log_dir}.")
    logging.info(f"Pretraining with the file {args['pretrained']}")

    for seed in seeds:
        print(f"Running with seed {seed}")
        logging.info(f"Running with seed {seed}")
        seed_everything(seed)

        train_loaders, val_loader = dataprep(args)

        baseline = False
        for i, train_loader in enumerate(train_loaders):
            output = {len(train_loader.dataset): {
                "baseline": 0,
                "pretrained": 0,
            }}
            for _ in range(2):
                print(f"Training on {len(train_loader.dataset)} ECGs and validation on {len(val_loader.dataset)} ECGs.")
                model = create_model(args, baseline=baseline)
                print(model.conv1.weight.requires_grad)
                model.to(device)
                lr = 1e-3 if baseline else 0.2
                numEpoch = epochs[i]

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                best_auc_test = Training.train(
                    model=model,
                    trainDataLoader=train_loader,
                    testDataLoader=val_loader,
                    numEpoch=numEpoch,
                    optimizer=optimizer,
                )

                logging.info(f'For seed {seed}, with {len(train_loader.dataset)} ECGs, {"Baseline" if baseline else "PreTrained"} model  BEST_AUC: {best_auc_test}')

                output[len(train_loader.dataset)]["baseline" if baseline else "pretrained"] = best_auc_test

                baseline = not baseline
            
            
            results[seed].append(output)
            
            
    print(results)
    with open(f'results/{results_file}.json', 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    now = time.time()
    main(args)
    print(f"Total Time: {time.time() - now} seconds")
    
