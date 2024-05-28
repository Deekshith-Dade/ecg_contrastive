from torch.utils.data import Dataset
import numpy as np
import torch as tch
import os
import json



class PreTrainECGDatasetLoader(Dataset):
    
    def __init__(self, augs=None, baseDir='', patients=[], normalize=True, normMethod='unitrange', rhythmType='Rhythm'):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []
        self.augs = augs

        
        
        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients

        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            for i in range(2):
                ecgId = str(patientInfo["ecgFileIds"][ecgIx])
                zeros = 5 - len(ecgId)
                ecgId = "0"*zeros+ ecgId
                self.fileList.append(os.path.join(patient,
                                            f'ecg_{ecgIx}',
                                            f'{ecgId}_{self.rhythmType}.npy'))
                self.patientLookup.append(f"{patient}_{i}")
            
    def __getitem__(self, item):
        if self.augs is None:
            print("Provide TwoCropsTransform object as aug")
            return 

        segment = self.patientLookup[item][-1]
        
        # patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        # patientInfo = json.load(open(patientInfoPath))
        # ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])

        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        

        ecgData = np.load(ecgPath)

        if(segment == '0'):
            ecgData = ecgData[:, 0:2500]
        else:
            ecgData = ecgData[:, 2500:]

        ecgs = tch.tensor(ecgData).float()
        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - tch.min(frame)) / (tch.max(frame) - tch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return (self.augs(ecgs), self.patientLookup[item][:-2])
    
    def __len__(self):
        return len(self.fileList)

class PreTrainECGDatasetLoaderV2(Dataset):
    
    def __init__(self, augs=None, baseDir='', patients=[], normalize=True, normMethod='unitrange', rhythmType='Rhythm'):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []
        self.augs = augs

        
        
        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients

        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["ecgFileIds"][ecgIx])
            zeros = 5 - len(ecgId)
            ecgId = "0"*zeros+ ecgId
            self.fileList.append(os.path.join(patient,
                                        f'ecg_{ecgIx}',
                                        f'{ecgId}_{self.rhythmType}.npy'))
            self.patientLookup.append(f"{patient}")
            
    def __getitem__(self, item):
        if self.augs is None:
            print("Provide TwoCropsTransform object as aug")
            return 

        # patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        # patientInfo = json.load(open(patientInfoPath))
        # ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])

        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        

        ecgData = np.load(ecgPath)

        ecgs = tch.tensor(ecgData).float()
        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - tch.min(frame)) / (tch.max(frame) - tch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return (self.augs(ecgs), self.patientLookup[item])
    
    def __len__(self):
        return len(self.fileList)


class PatientECGDatasetLoader(Dataset):
    
    def __init__(self, baseDir='', patients=[], normalize=True, normMethod='unitrange', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.fileList = []
        self.patientLookup = []

        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients
        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        if numECGstoFind == 'all':
            for pat in self.patients:
                self.findEcgs(pat, 'all')
        else:
            for pat in self.patients:
                self.findEcgs(pat, numECGstoFind)
    
    def findEcgs(self, patient, numberToFind=1):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['numberOfECGs']

        if(numberToFind == 1) | (numberOfEcgs == 1):
            for i in range(2):
                ecgId = str(patientInfo["ecgFileIds"][0])
                zeros = 5 - len(ecgId)
                ecgId = "0"*zeros+ ecgId
                self.fileList.append(os.path.join(patient,
                                    f'ecg_0',
                                    f'{ecgId}_{self.rhythmType}.npy'))
                self.patientLookup.append(f"{patient}_{i}")
        else:
            for ecgIx in range(numberOfEcgs):
                for i in range(2):
                    self.fileList.append(os.path.join(patient,
                                                f'ecg_{ecgIx}',
                                                f'{patientInfo["ecgFields"][ecgIx]}_{self.rhythmType}.npy'))
                    self.patientLookup.append(f"{patient}_{i}")
        
    
    def __getitem__(self, item):
        patient = self.patientLookup[item][:-2]
        segment = self.patientLookup[item][-1]

        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)
        if(segment == '0'):
            ecgData = ecgData[:, 0:2500]
        else:
            ecgData = ecgData[:, 2500:]

        ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])
        ecgs = tch.tensor(ecgData).float()

        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - tch.min(frame)) / (tch.max(frame) - tch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return ecgs, ejectionFraction
    
    def __len__(self):
        return len(self.fileList)


class ECGDatasetLoader(Dataset):
    
    def __init__(self,  baseDir='', patients=[], normalize=True, normMethod='0to1', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []

        
        
        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients

        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["ecgFileIds"][ecgIx])
            zeros = 5 - len(ecgId)
            ecgId = "0"*zeros+ ecgId
            self.fileList.append(os.path.join(patient,
                                        f'ecg_{ecgIx}',
                                        f'{ecgId}_{self.rhythmType}.npy'))
            self.patientLookup.append(patient)
            
    def __getitem__(self, item):
        
        patientInfoPath = os.path.join(self.baseDir, self.patientLookup[item], 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)

        ejectionFraction = tch.tensor(patientInfo['ejectionFraction'])
        ecgs = tch.tensor(ecgData).float().unsqueeze(0)

        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return ecgs, ejectionFraction
    
    def __len__(self):
        return len(self.fileList)
    
class ECG_Sex_DatasetLoader(Dataset):
    
    def __init__(self,  baseDir='', patients=[], normalize=True, normMethod='0to1', rhythmType='Rhythm', numECGstoFind=1):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.patientLookup = []

        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients
        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        self.fileList = []
        if len(patients) != 0:
            for pat in self.patients:
                self.findEcgs(pat)
        
    
    def findEcgs(self, patient):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs']
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["ecgFileIds"][ecgIx])
            label = str(patientInfo["gender"][ecgIx])
            zeros = 5 - len(ecgId)
            ecgId = "0"*zeros+ ecgId
            self.fileList.append(os.path.join(patient,
                                        f'ecg_{ecgIx}',
                                        f'{ecgId}_{self.rhythmType}.npy'))
            self.patientLookup.append((patient, label))
            
    def __getitem__(self, item):
        
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)
        label = self.patientLookup[item][1]
        gender = 0 if label == "Female" else 1
        gender = tch.tensor(gender).float()
        ecgs = tch.tensor(ecgData).float().unsqueeze(0)

        if self.normalize:
            if self.normMethod == '0to1':
                if not tch.allclose(ecgs, tch.zeros_like(ecgs)):
                    ecgs = ecgs - tch.min(ecgs)
                    ecgs = ecgs / tch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if tch.any(tch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return ecgs, gender
    
    def __len__(self):
        return len(self.fileList)