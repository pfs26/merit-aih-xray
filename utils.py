import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Safe style fallback to avoid seaborn-notebook errors
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        import seaborn as sns
        sns.set_theme()
    except ImportError:
        plt.style.use('default')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Indicate the path to the dataset
dirs = {
    'train' : 'data/train',
    'val' : 'data/val',
    'test' : 'data/test'
}

#Set of transformations to be applied to the images
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

#Return the predictions of the model on the dataloader
def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds

# Return how many predictions are correct
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Return confusion matrix as numpy array
def get_confmat(test_set, test_preds):
    y_pred = torch.argmax(test_preds, dim=1)
    return confusion_matrix(test_set.targets, y_pred)

# Denormalize and convert tensor image to numpy for visualization
def deprocess_image(image):
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image
