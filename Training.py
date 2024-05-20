import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bce_loss = nn.BCELoss()

def loss_bce(predictedVal, clinicalParam):
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


def train(model, trainDataLoader, testDataLoader, numEpoch, optimizer):
    print(f"Beginning Training for Network {model.__class__.__name__}")
    best_auc_test = 0.5

    for ep in range(numEpoch):
        print(f"Epoch {ep+1} of {numEpoch}")

        model.train()

        count = 0
        running_loss = 0.0

        for ecg, clinicalParam in trainDataLoader:
            print(f'Running through training batches {count} of {len(trainDataLoader)}', end='\r')

            count += 1
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                ecg = ecg.to(device)
                clinicalParam = clinicalParam.to(device).unsqueeze(1)
                predicted = model(ecg)
                loss = loss_bce(predicted, clinicalParam)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()
            
        batch_total_loss = running_loss / len(trainDataLoader)
        print()
        print(f"Batch Loss: {batch_total_loss}")

        print('Evalving Test')
        currTestLoss, allParams_test, allPredictions_test, _ = evaluate(model, testDataLoader, loss_bce)
        print('Evalving Train')
        currTrainLoss, allParams_train, allPredictions_train, _ = evaluate(model, trainDataLoader, loss_bce)
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
        

        plt.figure(1)
        fig, ax1 = plt.subplots(1, 2)

        print(f'Train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f}')
        ax1[0].plot(falsePos_train, truePos_train)
        ax1[0].set_title(f'ROC train, AUC: {auc_train:0.3f}')
        ax1[1].plot(falsePos_test, truePos_test)
        ax1[1].set_title(f'ROC Test, AUC: {auc_test:0.3f}')
        plt.suptitle(f'ROC curves train AUC: {auc_train:0.3f} test AUC: {auc_test:0.3f}')
        # plt.show()

    print(f"Best AUC Test: {best_auc_test}")
