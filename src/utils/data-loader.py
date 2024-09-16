import torch
from torch.utils.data import DataLoader
from src.preprocessing.data_pruning import prune_data
from src.preprocessing.data_synthesis import synthesize_data
from src.preprocessing.dimensionality_reduction import reduce_dimensions
from src.preprocessing.efficient_storage import load_from_hdf5

def create_data_loader(dataset, batch_size=32, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2  # Adjust based on your system
    )

def load_data(batch_size=32, apply_augmentations=True, apply_feature_selection=False, 
              apply_normalization=False, apply_pruning=False, apply_synthesis=False, 
              apply_dimensionality_reduction=False, storage_file=None):
    
    if storage_file:
        X, y = load_from_hdf5(storage_file)
    else:
        transforms = get_data_transforms() if apply_augmentations else None
        dataset = datasets.YourDataset(root='path/to/data', transform=transforms)
        X, y = dataset.features, dataset.labels

    if apply_feature_selection:
        X, _ = apply_feature_selection(X, y, k=10)

    if apply_pruning:
        X, _ = prune_data(X, y, threshold=0.01)
    
    if apply_synthesis:
        X, y = synthesize_data(X, y)
    
    if apply_dimensionality_reduction:
        X, _ = reduce_dimensions(X, n_components=50)
    
    if apply_normalization:
        X, _ = normalize_data(X)
    
    if storage_file:
        save_to_hdf5(storage_file, X, y)

    dataset = CustomDataset(X, y)  # Ensure CustomDataset handles preprocessed features

    return create_data_loader(dataset, batch_size=batch_size)
