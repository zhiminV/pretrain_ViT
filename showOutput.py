import matplotlib.pyplot as plt
from matplotlib import colors
from dataset_wrapper import test_loader, model

def show_inference(n_rows, features, label, prediction_function):
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    
    fig = plt.figure(figsize=(15, n_rows * 4))
    
    features = features.permute(0, 2, 3, 1)  # Change feature tensor to (batch, height, width, channels)
    prediction = prediction_function(features)

    for i in range(n_rows):
        plt.subplot(n_rows, 3, i * 3 + 1)
        plt.title("Previous day fire")
        plt.imshow(features[i, :, :, -1].cpu().numpy(), cmap=CMAP, norm=NORM)
        plt.axis('off')

        plt.subplot(n_rows, 3, i * 3 + 2)
        plt.title("True next day fire")
        plt.imshow(label[i, 0, :, :].cpu().numpy(), cmap=CMAP, norm=NORM)
        plt.axis('off')

        plt.subplot(n_rows, 3, i * 3 + 3)
        plt.title("Predicted next day fire")
        plt.imshow(prediction[i, 0, :, :], cmap=CMAP, norm=NORM)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_results.png')
    plt.show()

features, labels = next(iter(test_loader))
features_torch = features.permute(0, 3, 1, 2).float().to(device)
labels_torch = labels.permute(0, 3, 1, 2).float().to(device)
show_inference(5, features_torch, labels_torch, lambda x: torch.sigmoid(model(x.permute(0, 3, 1, 2))).detach().cpu().numpy())
