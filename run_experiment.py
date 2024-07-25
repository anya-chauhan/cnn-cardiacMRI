import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import cMRIDataset
from transforms import get_train_transforms, get_val_test_transforms, calculate_stats
from train_and_evaluate import train_and_evaluate

# Model parameters
MODEL_ARCHITECTURE = 'resnet50'  
LOSS_FUNCTION = 'huber'  
OPTIMIZER = 'adamw'  
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BATCH_SIZE = 16
EPOCHS = 100

def prepare_data(df, val_size=50, test_size=50):
    train_val, test = train_test_split(df, test_size=test_size, random_state=48)
    train, val = train_test_split(train_val, test_size=val_size, random_state=48)
    return train, val, test

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/labels.csv")[['idx', 'sphericity_index']]
    
    # Prepare the data
    train_pool, val, test = prepare_data(df, val_size=50, test_size=50)

    train_pool['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'

    # Define the training sizes to try
    train_sizes = [350]  # Modify as needed

    # Train and evaluate for each training size
    results = []
    for size in train_sizes:
        # Subsample the training data
        train_subset = train_pool.sample(n=size, random_state=24)

        # Combine subsampled train data with val and test data
        df_subset = pd.concat([train_subset, val, test])

        # Set up data for calculating mean and std
        raw_train_dataset = cMRIDataset(df_subset, split='train', transform=None)
        mean, std = calculate_stats(raw_train_dataset)
        print(f"Calculated mean: {mean}, std: {std}")

        # Create datasets with appropriate transforms
        train_dataset = cMRIDataset(df_subset, split='train', transform=get_train_transforms(mean, std))
        val_dataset = cMRIDataset(df_subset, split='val', transform=get_val_test_transforms(mean, std))
        test_dataset = cMRIDataset(df_subset, split='test', transform=get_val_test_transforms(mean, std))

        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        # Call the train_and_evaluate function
        result = train_and_evaluate(df_subset,
                                    MODEL_ARCHITECTURE, LOSS_FUNCTION, OPTIMIZER,
                                    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EPOCHS)

        # Add train size to the result dictionary
        result['train_size'] = size

        results.append(result)
        print(f"Training Size: {size}, R2 Score: {result['r2']:.4f}")

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('train_size_experiment_results.csv', index=False)

