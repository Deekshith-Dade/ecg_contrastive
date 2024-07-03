import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import wandb
import copy
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bce_loss = nn.BCELoss()

def loss_bce_lvef(predictedVal, clinicalParam):
    clinicalParam = (clinicalParam < 40.0).float()
    return bce_loss(predictedVal, clinicalParam)


def evaluate(network, dataloader, lossFun):
    network.eval()
    with torch.no_grad():
        running_loss = 0.
        allParams = torch.empty(0).to(device)
        allPredictions=torch.empty(0).to(device)
        allNoiseVals = np.empty((0, 8))

        for ecg, clinicalParam in dataloader:
            ecg = ecg.to(device)
            clinicalParam = clinicalParam.to(device).unsqueeze(1)
            predictedVal = network(ecg)
            lossVal = lossFun(predictedVal, clinicalParam)
            running_loss += lossVal.item()
            allParams = torch.cat((allParams, clinicalParam.squeeze()))
            allPredictions = torch.cat((allPredictions, predictedVal.squeeze()))

        running_loss = running_loss/len(dataloader)
    return running_loss, allParams, allPredictions, allNoiseVals


def train(model, trainDataLoader, testDataLoader, numEpoch, optimizer, modelSaveDir, modelName, logToWandB=False ):
    print(f"Beginning Training for Network {model.__class__.__name__}")
    best_auc_test = 0.5
    best_acc = 0.5

    for ep in range(numEpoch):
        print(f"Epoch {ep+1} of {numEpoch}")

        model.train()

        count = 0
        running_loss = 0.0

        for ecg, clinicalParam in trainDataLoader:
            print(f'Running through training batches {count+1} of {len(trainDataLoader)}', end='\r')

            count += 1
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                ecg = ecg.to(device)
                clinicalParam = clinicalParam.to(device).unsqueeze(1)
                predicted = model(ecg)
                loss = loss_bce_lvef(predicted, clinicalParam)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()
            
        batch_total_loss = running_loss / len(trainDataLoader)
        print()
        print(f"Batch Loss: {batch_total_loss}")

        print('Evalving Test')
        currTestLoss, allParams_test, allPredictions_test, _ = evaluate(model, testDataLoader, loss_bce_lvef)
        print('Evalving Train')
        currTrainLoss, allParams_train, allPredictions_train, _ = evaluate(model, trainDataLoader, loss_bce_lvef)
        print(f"Train Loss: {currTrainLoss} \n Test Loss: {currTestLoss}")

        allParams_train = (allParams_train.clone().detach().cpu() < 40.0).long().numpy()
        allPredictions_train = allPredictions_train.clone().detach().cpu().numpy()

        allParams_test = (allParams_test.clone().detach().cpu() < 40.0).long().numpy()
        allPredictions_test = allPredictions_test.clone().detach().cpu().numpy()

        falsePos_train, truePos_train, _ = metrics.roc_curve(allParams_train, allPredictions_train)
        falsePos_test, truePos_test, _ = metrics.roc_curve(allParams_test, allPredictions_test)
        auc_train = metrics.roc_auc_score(allParams_train, allPredictions_train)
        auc_test = metrics.roc_auc_score(allParams_test, allPredictions_test)

        if auc_test > best_auc_test:
            best_auc_test = auc_test

            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model , os.path.join(modelSaveDir, f"{modelName}_best.pth"))

        
        if acc_test > best_acc:
            best_acc = acc_test
        
        precision, recall, thresholds = metrics.precision_recall_curve(allParams_test, allPredictions_test)
        denominator = recall+precision
        if np.any(np.isclose(denominator,[0.0])):
            print('\nSome precision+recall were zero. Setting to 1.\n')
            denominator[np.isclose(denominator,[0.0])] = 1
        
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores[np.isnan(f1_scores)] = 0
        maxIx = np.argmax(f1_scores)

        f1_score_test_max = f1_scores[maxIx]
        thresholdForMax = thresholds[maxIx]

        acc_test_f1max = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>thresholdForMax).astype('float'))
        acc_train_f1max = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>thresholdForMax).astype('float'))

        acc_test = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>0.5).astype('float'))
        acc_train = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>0.5).astype('float'))

        print(f'Weighted Acc at 50% cutoff: {acc_train:.4f} train {acc_test:.4f} test')

        if logToWandB:
            plt.figure(1)
            fig, ax1 = plt.subplots(1, 2)

            print(f'Train AUC: {auc_train:0.6f} test AUC: {auc_test:0.6f}')
            ax1[0].plot(falsePos_train, truePos_train)
            ax1[0].set_title(f'ROC train, AUC: {auc_train:0.3f}')
            ax1[1].plot(falsePos_test, truePos_test)
            ax1[1].set_title(f'ROC Test, AUC: {auc_test:0.3f}')
            plt.suptitle(f'ROC curves train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f} @ Epoch {ep+1} of {numEpoch}')
            
            print(f"Figures Made")
            logDict = {
                'Epoch': ep,
                'Training Loss': currTrainLoss,
                'Test Loss': currTestLoss,
                'auc test': auc_test,
                'acc test': acc_test,
                'acc test f1max': acc_test_f1max,
                'f1 max test': f1_score_test_max,
                'max f1 threshold': thresholdForMax,
                'auc train': auc_train,
                'acc train': acc_train,
                'acc train f1max': acc_train_f1max,
                'ROCs individual': plt
            }
            print(f"Log Dict Created and Logging to WandB")
            wandb.log(logDict)

    print(f"Best AUC Test: {best_auc_test}")
    return best_auc_test , best_acc

def loss_bce_kcl(predictedVal,clinicalParam,lossParams):
	clinicalParam = ((clinicalParam <= lossParams['highThresh']) * (clinicalParam >= lossParams['lowThresh'])).float()
	return bce_loss(predictedVal,clinicalParam)

def evaluate_balanced(network,dataLoaders,lossFun,lossParams,leads,lookForFig=False):
    network.eval()
    plt.figure(2)
    fig, ax1 = plt.subplots(8,2, figsize=(4*15, 4*8*2.5))
    pltCol = 0
					
    with torch.no_grad():
        running_loss = 0.
		
        allParams = torch.empty(0).to(device)
        allPredictions = torch.empty(0).to(device)
        for dataLoader in dataLoaders:
            for ecg, clinicalParam in dataLoader:
                ecg = ecg[:,leads,:].to(device)
                clinicalParam = clinicalParam.to(device).unsqueeze(1) 
                predictedVal = network(ecg)
                lossVal = lossFun(predictedVal,clinicalParam,lossParams)
                running_loss += lossVal.item()

                allParams = torch.cat((allParams, clinicalParam.squeeze()) )
                allPredictions = torch.cat((allPredictions, predictedVal.squeeze()))
                if lookForFig:
                    binaryParams = ((clinicalParam <= lossParams['highThresh']) * (clinicalParam >= lossParams['lowThresh'])).float()
                    agreement = torch.abs(binaryParams.squeeze()-predictedVal.squeeze())
                    ecgIx = torch.argmax(agreement)
                    disagree_kcl = clinicalParam[ecgIx,...]
                    disagree_pred = allPredictions[ecgIx,...]
                    for lead in range(8):
                        ax1[lead,pltCol].plot(ecg[ecgIx,lead,:].detach().clone().squeeze().cpu().numpy(),'k')
                    ecgIx = torch.argmin(agreement)
                    agree_kcl = clinicalParam[ecgIx,...]
                    agree_pred = allPredictions[ecgIx,...]
                    for lead in range(8):
                        ax1[lead,pltCol+1].plot(ecg[ecgIx,lead,:].detach().clone().squeeze().cpu().numpy(),'k')
                    lookForFig = False
                    ax1[0,pltCol].text(0,100, f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.')
                    ax1[0,pltCol+1].text(0, 100,f'Agree: {agree_kcl.item()}, pred: {agree_pred}.')
                    fig.suptitle(f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.\nAgree: {agree_kcl.item()}, pred: {agree_pred}.', fontsize=50, y=0.95)
                    print(f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.\nAgree: {agree_kcl.item()}, pred: {agree_pred}.')		
        totalDataLoaderLens = [len(d) for d in dataLoaders]
        running_loss = running_loss/sum(totalDataLoaderLens)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
        plot_margin = 0.25

        # x0, x1, y0, y1 = plt.axis()
        # plt.axis((x0 - plot_margin,
        #         x1 + plot_margin,
        #         y0 - plot_margin,
        #         y1 + plot_margin))

	
        return running_loss, allParams, allPredictions, fig


def trainNetwork_balancedClassification(network, trainDataLoader_normals, trainDataLoader_abnormals, testDataLoader, numEpoch, optimizer, lossFun, lossParams, label, logToWandB, leads):
    print(f"Beginning Training for Network {network.__class__.__name__}")
    prevTrainingLoss = 0.0
    bestEvalMetric_test = 0.5
    best_acc = 0.5
    maxBatches = max(len(trainDataLoader_normals), len(trainDataLoader_abnormals))

    for ep in range(numEpoch):
        print(f"Epoch {ep+1} of {numEpoch}")
        running_loss = 0.0
        network.train()

        iter_normal = iter(trainDataLoader_normals)
        iter_abnormal = iter(trainDataLoader_abnormals)

        for batchIx in range(maxBatches):
            try:
                ecg_normal, clinicalParam_normal = next(iter_normal)
            except:
                iter_normal = iter(trainDataLoader_normals)
                ecg_normal, clinicalParam_normal = next(iter_normal)
            
            try:
                ecg_abnormal, clinicalParam_abnormal = next(iter_abnormal)
            except:
                iter_abnormal = iter(trainDataLoader_abnormals)
                ecg_abnormal, clinicalParam_abnormal = next(iter_abnormal)
            
            numNormal = ecg_normal.shape[0]
            numAbnormal = ecg_abnormal.shape[0]

            ecg = torch.empty((numNormal+numAbnormal, *list(ecg_normal.shape[1:])))
            clincalParam = torch.empty((numNormal+numAbnormal, *list(clinicalParam_normal.shape[1:])))

            shuffleIxs = torch.randperm(numNormal+numAbnormal)
            normIxs = shuffleIxs[:numNormal]
            abnormIxs = shuffleIxs[numNormal:]

            ecg[normIxs,...] = ecg_normal
            clincalParam[normIxs,...] = clinicalParam_normal
            ecg[abnormIxs,...] = ecg_abnormal
            clincalParam[abnormIxs,...] = clinicalParam_abnormal

            print(f'Running through training batches {batchIx} of {maxBatches}. Input Size {ecg.shape}           ',end='\r')

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                ecg = ecg[:,leads,:].to(device)
                clinicalParam = clincalParam.to(device).unsqueeze(1)

                predictedVal = network(ecg)
                lossVal = lossFun(predictedVal, clinicalParam, lossParams)
                lossVal.backward()
                optimizer.step()
                running_loss = running_loss + lossVal
            
        currTrainingLoss = running_loss/(len(trainDataLoader_normals.dataset)+len(trainDataLoader_abnormals.dataset))

        print(f'Using lead {leads}. Running through training batches end of {maxBatches}     ')
		#print useful messages
        print(f"Epoch {ep+1} train loss {currTrainingLoss}, Diff {currTrainingLoss - prevTrainingLoss}")
        prevTrainingLoss = currTrainingLoss
        print('Evaling test')
        currTestLoss, allParams_test, allPredictions_test, ecgFig = evaluate_balanced(network,[testDataLoader],lossFun,lossParams,leads,lookForFig=True)
        print('Evaling train')
        currTrainLoss, allParams_train, allPredictions_train, _ = evaluate_balanced(network,[trainDataLoader_normals,trainDataLoader_abnormals],
																					lossFun,lossParams,leads)
        print(f"train loss: {currTrainLoss}, val loss: {currTestLoss}")

        #process results
        allParams_train = allParams_train.clone().detach().cpu().numpy()
        allPredictions_train = allPredictions_train.clone().detach().cpu().numpy()
        allParams_test = allParams_test.clone().detach().cpu().numpy()
        allPredictions_test = allPredictions_test.clone().detach().cpu().numpy()

        #convert params to binary
        allParams_train = ((allParams_train <= lossParams['highThresh']) * (allParams_train >= lossParams['lowThresh'])).astype(float)
        allParams_test = ((allParams_test <= lossParams['highThresh']) * (allParams_test >= lossParams['lowThresh'])).astype(float)
        print(f'For train, {sum(allParams_train)} normals, for test {sum(allParams_test)} normals')

        #get roc curve and auc
        print('Calculating ROC and other metrics')
        falsePos_train, truePos_train, thresholds_train = metrics.roc_curve(allParams_train,allPredictions_train)
        falsePos_test, truePos_test, thresholds_test = metrics.roc_curve(allParams_test,allPredictions_test)
		
        evalMetric_train = metrics.roc_auc_score(allParams_train, allPredictions_train)
        evalMetric_test = metrics.roc_auc_score(allParams_test, allPredictions_test)

        if evalMetric_test > bestEvalMetric_test:
             bestEvalMetric_test = evalMetric_test
        
        
        
        
        precision, recall, thresholds = metrics.precision_recall_curve(allParams_test, allPredictions_test)
        denominator = recall+precision
        if np.any(np.isclose(denominator,[0.0])):
            print('\nSome precision+recall were zero. Setting to 1.\n')
            denominator[np.isclose(denominator,[0.0])] = 1

        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores[np.isnan(f1_scores)] = 0
        maxIx = np.argmax(f1_scores)

        f1_score_test_max = f1_scores[maxIx]
        thresholdForMax = thresholds[maxIx]

        acc_test_f1max = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>thresholdForMax).astype('float'))
        acc_train_f1max = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>thresholdForMax).astype('float'))

        acc_test = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>0.5).astype('float'))
        acc_train = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>0.5).astype('float'))
        
        if acc_test > best_acc:
            best_acc = acc_test
        
        print(f'Weighted Acc at 50% cutoff: {acc_train:.4f} train {acc_test:.4f} test')

        if logToWandB:
            print('Logging to wandb')
            plt.figure(1)
            fig,ax1 = plt.subplots(1,2)

            print(f'train score: {evalMetric_train:0.4f} test score: {evalMetric_test:0.4f}')
            ax1[0].plot(falsePos_train,truePos_train)
            ax1[0].set_title(f'ROC train, AUC: {evalMetric_train:0.3f}')
            ax1[1].plot(falsePos_test,truePos_test)
            ax1[1].set_title(f'ROC test, AUC: {evalMetric_test:0.3f}')
            plt.suptitle(f'ROC curves train AUC: {evalMetric_train:0.3f} test AUC: {evalMetric_test:0.3f}')

            print(f"Figures Made")
            logDict = {
                 'Epoch': ep,
                 'Training Loss': currTrainingLoss,
                 'Test Loss': currTestLoss,
                 'auc test': evalMetric_test,
                 'acc test': acc_test,
                 'acc test f1max': acc_test_f1max,
                 'f1 max test': f1_score_test_max,
                 'max f1 threshold': thresholdForMax,
                 'auc train': evalMetric_train,
                 'acc train': acc_train,
                 'acc train f1max': acc_train_f1max,
                 'ROCs individual': plt,
                 'ECGs Examples': ecgFig
            }
            print(f"Log Dict Created and Logging to WandB")
            wandb.log(logDict)
            plt.close(ecgFig)

    print(f"Best AUC Test: {bestEvalMetric_test}")
    return bestEvalMetric_test, best_acc














