import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import deprocess_image
from sklearn.metrics import ConfusionMatrixDisplay

    
def plot_confmat(confusion_matrix, labels, filename):

    os.makedirs("outputs", exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix", fontsize=20)

    fig.savefig(f"outputs/{filename}", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
def plot_gradcam(image, dense_cam, filename):
    
    image = deprocess_image(image)
    name_dict = {
        'Original Image': image,
        'GradCAM (DenseNet-121)': apply_mask(image, dense_cam)
    }

    plt.style.use('seaborn-notebook')
    fig = plt.figure(figsize=(8, 4))

    
    for i, (name, img) in enumerate(name_dict.items()):
        ax = fig.add_subplot(1, 2, i+1, xticks=[], yticks=[])
        if i:
            img = img[:, :, ::-1]
        ax.imshow(img)
        ax.set_xlabel(name)

    fig.suptitle(
        'Localization with Gradient based Class Activation Maps', fontsize=14
    )
    plt.tight_layout()
    fig.savefig('outputs/' + filename +'.png')
    plt.show()
    plt.close()
    
    
def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)