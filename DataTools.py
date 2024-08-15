from torch.utils.data import Dataset
import numpy as np
import torch 
import os
import json
import torch.nn.functional as F


class DataLoaderError(Exception):
    pass


class PreTrain_1M_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],patientIds=[],augs=None, normalize =False, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=False,
				 mismatchFix='Pad',randomCrop=True,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		self.normalize = normalize
		self.normMethod = normMethod
		self.augs = augs
		self.ecgs = ecgs
		self.patientIds = patientIds
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		ecgs = torch.tensor(ecgData).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
			
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return self.augs(ecgs), str(self.patientIds[item])

	def __len__(self):
		return len(self.ecgs)


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
        # ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])

        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        

        ecgData = np.load(ecgPath)

        if(segment == '0'):
            ecgData = ecgData[:, 0:2500]
        else:
            ecgData = ecgData[:, 2500:]

        ecgs = torch.tensor(ecgData).float()
        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - torch.min(frame)) / (torch.max(frame) - torch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
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
        # ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])

        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        

        ecgData = np.load(ecgPath)

        ecgs = torch.tensor(ecgData).float()
        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - torch.min(frame)) / (torch.max(frame) - torch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
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

        ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])
        ecgs = torch.tensor(ecgData).float()

        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - torch.min(frame)) / (torch.max(frame) - torch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
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

        ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])
        ecgs = torch.tensor(ecgData).float().unsqueeze(0)

        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
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
        gender = torch.tensor(gender).float()
        ecgs = torch.tensor(ecgData).float().unsqueeze(0)

        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
            
        
        return ecgs, gender
    
    def __len__(self):
        return len(self.fileList)

class ECG_KCL_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],kclVals=[],normalize =True, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=True,
				 mismatchFix='Pad',randomCrop=False,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.kclVals = kclVals
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		kclVal = torch.tensor(self.kclVals[item])
		ecgs = torch.tensor(ecgData).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
			
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		return ecgs, kclVal

	def __len__(self):
		return len(self.ecgs)

class ECG_Genetics_Datasetloader(Dataset):
    def __init__(self, dataDir, patientIds, geneticResults, numECGsToFind='all', normalize=False, normMethod='0to1', rhythmType='Rhythm', allowMismatchTime=True, mismatchFix='Pad', randomCrop=False, cropSize=2500, expectedTime=5000):
        self.ecgs = []
        self.dataDir = dataDir
        self.normalize = normalize
        self.normMethod = normMethod
        self.rhythmType=rhythmType
        self.randomCrop = randomCrop
        self.cropSize = cropSize
        self.patients = patientIds
        self.geneticResults = geneticResults
        self.numECGsToFind = numECGsToFind
        self.ecgCounts = []
        self.fileList = []
        self.geneticVals = []
        self.accGeneticResults = ["positive", "negative", "uncertain"]

        for i, patient in enumerate(self.patients):
            if geneticResults[i] in self.accGeneticResults:
                count = self.findEcgs(str(patient), self.geneticResults[i])
                self.ecgCounts.append((patient, count))

    
    def findEcgs(self, patient, geneticResult):
        patientInfoPath = os.path.join(self.dataDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['validECGs'] if self.numECGsToFind == 'all' else self.numECGsToFind
        for ecgIx in range(numberOfEcgs):
            ecgId = str(patientInfo["validFiles"][ecgIx]).replace('.xml', f'_{self.rhythmType}.npy')
            self.fileList.append(os.path.join(patient,f'ecg_{ecgIx}',ecgId))
            self.geneticVals.append(geneticResult)
        return numberOfEcgs

    
    def __getitem__(self, item):
        ecgPath = os.path.join(self.dataDir, self.fileList[item])
        
        ecgData = np.load(ecgPath)
        ecg = torch.tensor(ecgData).float()

        if self.randomCrop:
            startIx = 0
            if ecg.shape[-1] > self.cropSize:
                startIx = torch.randint(ecg.shape[-1]-self.cropSize, (1,))
            ecg = ecg[..., startIx:startIx+self.cropSize]

        return ecg, self.geneticVals[item]

    def __len__(self):
        return len(self.fileList)
