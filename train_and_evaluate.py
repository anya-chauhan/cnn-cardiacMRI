import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from dataset import cMRIDataset
from transforms import get_train_transforms, get_val_test_transforms, calculate_stats
from model import CNN

def train_and_evaluate(df_subset, model_architecture, loss_function, optimizer, learning_rate, weight_decay, batch_size, epochs):
    # Set up data
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

    dataloaders = {s: DataLoader(datasets[s], batch_size=batch_size, drop_last=True, shuffle=(s=='train'))
                   for s in ['train', 'val', 'test']}

    # Set up model and training
    cnn = CNN(model_architecture, loss_function, optimizer, learning_rate, weight_decay)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    early_stop_callback = EarlyStopping(
      monitor='val_loss',
      min_delta=0.00,
      patience=20,
      verbose=False,
      mode='min'
    )
    trainer = Trainer(logger=logger, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=epochs, log_every_n_steps=10)

    # Train the model
    trainer.fit(model=cnn, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'])

    # Load best model and evaluate
    best_model_path = checkpoint_callback.best_model_path
    model = CNN.load_from_checkpoint(best_model_path, model_architecture=model_architecture,
                                     loss_function=loss_function, optimizer=optimizer, learning_rate=learning_rate,
                                     weight_decay=weight_decay)
    model.eval()

    # Evaluate on test set
    test_results = trainer.test(model, dataloaders=dataloaders['test'])
    test_loss = test_results[0]['test_loss']

    # Get predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds = []
    all_real = []
    model = model.to(device)
    for mri, si in dataloaders['test']:
        mri = mri.to(device)
        preds = model.predict_si(mri)
        all_preds.extend(preds.cpu().numpy())
        all_real.extend(si.numpy())

    preds = np.array(all_preds)
    real = np.array(all_real)

    # Calculate metrics
    mse = mean_squared_error(real, preds)
    pearson_r, _ = pearsonr(real, preds)
    r2 = pearson_r**2

    # Save results
    results = {
        'model_architecture': model_architecture,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'test_loss': test_loss,
        'mse': mse,
        'r2': r2,
        'best_model_path': best_model_path
    }

    # Append results to CSV
    results_df = pd.DataFrame([results])
    if os.path.exists('cnn_results.csv'):
        results_df.to_csv('cnn_results.csv', mode='a', header=False, index=False)
    else:
        results_df.to_csv('cnn_results.csv', index=False)

    return results
