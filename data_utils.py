import numpy as np
import DataTools
import pickle
import random
import torch
import pandas as pd

def dataprepKCL(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/AllClinicalECGs/'
    normEcgs = False

    with open('kcl_patient_splits/test_patients.pkl','rb') as f:
        testECGs = pickle.load(f)
    with open('kcl_patient_splits/train_normal_patients.pkl','rb') as f:
        trainECGs_normal = pickle.load(f)
    with open('kcl_patient_splits/train_abnormal_patients.pkl','rb') as f:
        trainECGs_abnormal = pickle.load(f)

    trainECGs_normal_count = len(trainECGs_normal)
    trainECGs_abnormal_count = len(trainECGs_abnormal)

    finetuning_ratios = [0.01, 0.05, 0.1, 0.5, 1.0]
    num_finetuning = [[int(trainECGs_normal_count * ratio), int(trainECGs_abnormal_count * ratio)] for ratio in finetuning_ratios]

    train_loaders = []
    dataset_lengths = []
    for i in num_finetuning:
        finetuning_patients_normal = trainECGs_normal.sample(n=i[0])
        finetuning_patients_abnormal = trainECGs_abnormal.sample(n=i[1])

        trainData_normal_dataset = DataTools.ECG_KCL_Datasetloader(baseDir=dataDir+'pythonData/', 
                                                                    ecgs=finetuning_patients_normal['ECGFile'].tolist(),
                                                                    kclVals=finetuning_patients_normal['KCLVal'].tolist(),
                                                                    normalize=normEcgs,
                                                                    allowMismatchTime=False,
                                                                    randomCrop=True)
        trainData_abnormal_dataset = DataTools.ECG_KCL_Datasetloader(baseDir=dataDir+'pythonData/', 
                                                                    ecgs=finetuning_patients_abnormal['ECGFile'].tolist(),
                                                                    kclVals=finetuning_patients_abnormal['KCLVal'].tolist(),
                                                                    normalize=normEcgs,
                                                                    allowMismatchTime=False,
                                                                    randomCrop=True)

        trainData_normal_loader = torch.utils.data.DataLoader(trainData_normal_dataset,shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        trainData_abnormal_loader = torch.utils.data.DataLoader(trainData_abnormal_dataset,shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        dataset_lengths.append([len(trainData_normal_dataset), len(trainData_abnormal_dataset)])
        train_loaders.append([trainData_normal_loader, trainData_abnormal_loader])
    
    test_dataset = DataTools.ECG_KCL_Datasetloader(baseDir=dataDir+'pythonData/',
                                                   ecgs=testECGs['ECGFile'].tolist(),
                                                   kclVals=testECGs['KCLVal'].tolist(),
                                                   normalize=normEcgs,
                                                   allowMismatchTime=False,
                                                   randomCrop=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Preparing Finetuning with {dataset_lengths} number of ECGs and validation with {len(test_dataset)} number of ECGs")

    return train_loaders, test_loader
        
def dataprepLVEF(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = False

    print("Preparing Data For Finetuning")
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)
    
    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)
    
    num_classification_patients = len(pre_train_patients)
    finetuning_ratios =  args.finetuning_ratios
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        train_loaders.append(loader)
        dataset_lengths.append(len(dataset))
    
    validation_dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Preparing Finetuning with {dataset_lengths} number of ECGs and validation with {len(validation_dataset)} number of ECGs")

    return train_loaders, val_loader

def splitKCLPatients(seed):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/AllClinicalECGs/'
    timeCutoff = 3600 #seconds
    lowerCutoff = 1800 #seconds

    print("Finding Patients")
    kclCohort = np.load(dataDir+'kclCohort_v1.npy',allow_pickle=True)
    data_types = {
    'DeltaTime': float,   
    'KCLVal': float,    
    'ECGFile': str,     
    'PatId': int,       
    'KCLTest': str      
    }

    kclCohort = pd.DataFrame(kclCohort,columns=['DeltaTime','KCLVal','ECGFile','PatId','KCLTest']) 
    for key in data_types.keys():
        kclCohort[key] = kclCohort[key].astype(data_types[key])
    #remove those above cutoff
    kclCohort = kclCohort[kclCohort['DeltaTime']<=timeCutoff]
    kclCohort = kclCohort[kclCohort['DeltaTime']>lowerCutoff]#remove those below lower cutoff
    #remove nans
    kclCohort = kclCohort.dropna(subset=['DeltaTime']) 
    kclCohort = kclCohort.dropna(subset=['KCLVal']) 
    #kclCohort = kclCohort[0:2000]#jsut for testing
    #for each ECG, keep only the ECG-KCL pair witht he shortest delta time
    ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
    kclCohort = kclCohort.loc[ix]

    numPatients = len(np.unique(kclCohort['PatId']))

    print('setting up train/val split')
    numTest = int(0.1 * numPatients)
    numTrain = numPatients - numTest
    assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
    patientIds = list(np.unique(kclCohort['PatId']))
    random.Random(seed).shuffle(patientIds)

    trainPatientInds = patientIds[:numTrain]
    testPatientInds = patientIds[numTrain:numTest + numTrain]
    trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
    testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]

    perNetLossParams = dict(learningRate = 1e-3,highThresh = 5, lowThresh=4 ,type = 'binary cross entropy')

    
    trainECGs_normal = trainECGs[(trainECGs['KCLVal']>=perNetLossParams['lowThresh']) & (trainECGs['KCLVal']<=perNetLossParams['highThresh'])]
    trainECGs_abnormal = trainECGs[(trainECGs['KCLVal']<perNetLossParams['lowThresh']) | (trainECGs['KCLVal']>perNetLossParams['highThresh'])]

    #any additional processing
    #for this trial, only normal vs high. Remove the lows
    trainECGs_abnormal = trainECGs_abnormal[trainECGs_abnormal['KCLVal']>perNetLossParams['lowThresh']]
    testECGs = testECGs[testECGs['KCLVal']>perNetLossParams['lowThresh']]

    with open('kcl_patient_splits/train_normal_patients.pkl', 'wb') as file:
        pickle.dump(trainECGs_normal, file)
    
    with open('kcl_patient_splits/train_abnormal_patients.pkl', 'wb') as file:
        pickle.dump(trainECGs_abnormal, file)
    
    with open('kcl_patient_splits/test_patients.pkl', 'wb') as file:
        pickle.dump(testECGs, file)

    print(f'Found {len(testECGs)} tests and {len(trainECGs)} trains. In training, {len(trainECGs_normal)} are normal, {len(trainECGs_abnormal)} are abnormal')

def splitPatientsLVEF(seed):
    baseDir = ''
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    modelDir = ''
    normEcgs = False

    # Loading Data
    print('Finding Patients')
    allData = DataTools.PreTrainECGDatasetLoader(baseDir=dataDir, normalize=normEcgs)
    patientIds = np.array(allData.patients)
    numPatients = patientIds.shape[0]

    # Data
    pre_train_split_ratio = 0.9
    num_pre_train = int(pre_train_split_ratio * numPatients)
    num_validation = numPatients - num_pre_train

    patientInds = list(range(numPatients))
    random.Random(seed).shuffle(patientInds)

    pre_train_patient_indices = patientInds[:num_pre_train]
    validation_patient_indices = patientInds[num_pre_train:num_pre_train + num_validation]

    pre_train_patients = patientIds[pre_train_patient_indices].squeeze()
    validation_patients = patientIds[validation_patient_indices].squeeze()

    with open('patient_splits/pre_train_patients.pkl', 'wb') as file:
        pickle.dump(pre_train_patients, file)
    with open('patient_splits/validation_patients.pkl', 'wb') as file:
        pickle.dump(validation_patients, file)
    print(f"Out of Total {numPatients} Splitting {len(pre_train_patients)} for pre-train and finetuning, {len(validation_patients)} for validation")


