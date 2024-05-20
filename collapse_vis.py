import torch as tch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import torch as tch

import DataTools
import Networks

device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
gpuIds = list(range(tch.cuda.device_count()))

args = dict(
    batch_size=500,
    pretrained="runs/May19_20-36-19_cibcgpu4/checkpoint_0050.pth.tar",
    perplexity=50,
)

def main(args):
    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)

    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=pre_train_patients.tolist(), normalize=False)

    dataloader = tch.utils.data.DataLoader(
    dataset,
    batch_size=args["batch_size"],
    num_workers=32,
    shuffle=False,
    )

    model = Networks.BaselineConvNet(classification=False, avg_embeddings=True)
    model.finalLayer = nn.Identity()

    checkpoint = tch.load(args["pretrained"], map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.") and not k.startswith(
            "module.finalLayer."
        ):
            # remove prefix
            state_dict[k[len("module.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]


    msg = model.load_state_dict(state_dict, strict=True)
    print(f"There are {len(msg.missing_keys)} missing keys")
    
    model = tch.nn.DataParallel(model, device_ids=gpuIds)   
    print(model)
    model.to(device)
    model.eval()

    outputs = None
    labels = None
    for i, (ecg, clinicalParam) in enumerate(dataloader):
        
        print(f"Running through batch {i} of {len(dataloader)}", end='\r')
        ecg = ecg.to(device)
        out = model(ecg).detach().cpu()
        if outputs is None:
            outputs = out
        else:
            outputs = tch.cat((outputs, out), dim=0)
        if labels is None:
            labels = clinicalParam
        else:
            labels = tch.cat((labels, clinicalParam), dim=0)

    final = outputs.squeeze(1).cpu().numpy()
    print(final.shape)

    print("Running t-SNE")
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=args["perplexity"], learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(final)

    print("Plotting")
    plt.figure(figsize=(30, 18))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, alpha=0.5, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label value')

    # for i, label in enumerate(labels):
    #     plt.text(tsne_results[i, 0], tsne_results[i, 1], str(int(label.item())))

    plt.title(f"t-SNE visualization of model outputs with file {args['pretrained']}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f"pictures/collapse_output_{args['pretrained'].split('/')[1]}.png")
    print("Saved output.png")

if __name__ == "__main__":
    main(args)