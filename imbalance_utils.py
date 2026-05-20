#Module to handle imbalance in the dataset
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import ConcatDataset
import utils
from torch.utils.data import ConcatDataset, Subset


# Funció per calcular els pesos de les classes i crear un sampler per a l'entrenament
def get_class_weights(train_set):
    """
    Function that computes inverse-frequency class weights for a given training dataset.
    """
    targets=train_set.targets

    class_counts = torch.tensor(
        [targets.count(i) for i in range(len(train_set.classes))],
        dtype=torch.float
    )
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    return class_weights.to(utils.device)
  
#Funció que realiza el oversampling a la classe TB en aquest cas haurem multiplicar per 5 la classe TB per igualar el nombre d'exemples amb la classe No_TB

def oversampling_TB(train_set):
    """
    Function that performs oversampling of the TB class in the training dataset.
    """
    # 1. Convertimos los targets a tensor para operar fácilmente
    targets = torch.tensor(train_set.targets)
    class_counts = torch.bincount(targets)
    
    # Asumimos que 'Tuberculosis' es la clase 1 según tu lista class_names
    tb_class_idx = 1 
    
    majority_count = class_counts.max().item()
    tb_count = class_counts[tb_class_idx].item()
    
    # Calcular cuántas veces debemos repetir la clase minoritaria
    times_to_repeat = majority_count // tb_count
    
    # CORRECCIÓN: Buscamos explícitamente los índices de la clase TB (1)
    minority_indices = (targets == tb_class_idx).nonzero(as_tuple=True)[0].tolist()
    minority_subset = Subset(train_set, minority_indices)

    # Concatenamos el dataset original con las copias de la clase TB
    duplicated = ConcatDataset([train_set] + [minority_subset] * (times_to_repeat - 1))
    
    # CORRECCIÓN EXTRA: Reconstruimos el atributo .targets para el nuevo dataset
    # Esto evita que rompas funciones de graficado o conteo posteriores
    tb_targets = [tb_class_idx] * len(minority_indices)
    duplicated.targets = train_set.targets + tb_targets * (times_to_repeat - 1)
    
    return duplicated

if __name__=='__main__':
    class_names = ['Normal', 'Tuberculosis']

    train_set = datasets.ImageFolder(root=utils.dirs['train'], transform=transforms.ToTensor())

    targets_original = train_set.targets
    counts_original = [targets_original.count(i) for i in range(len(class_names))]
    print("--- Distribución Original ---")
    for name, count in zip(class_names, counts_original):
        print(f"{name}: {count} imágenes")
    print(f"Total original: {len(train_set)} imágenes\n")


    train_set_balanced = oversampling_TB(train_set)


    targets_balanced = train_set_balanced.targets
    counts_balanced = [targets_balanced.count(i) for i in range(len(class_names))]
    
    print("\n--- Distribución Final (Con Oversampling) ---")
    for name, count in zip(class_names, counts_balanced):
        print(f"{name}: {count} imágenes")
    print(f"Total nuevo: {len(train_set_balanced)} imágenes")
