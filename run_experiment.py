import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

        # Call the train_and_evaluate function
        result = train_and_evaluate(df_subset,
                                    MODEL_ARCHITECTURE, LOSS_FUNCTION, OPTIMIZER,
                                    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EPOCHS)

        # Add train size to the result dictionary
        result['train_size'] = size

        results.append(result)
        print(f"Training Size: {size}, R2 Score: {result['r2']:.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot([r['train_size'] for r in results], [r['r2'] for r in results], marker='o', label='R2 Score')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Score')
    plt.title('R2 Score vs Number of Training Examples')
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_pearson_vs_train_size.png')
    plt.close()

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('train_size_experiment_results.csv', index=False)
