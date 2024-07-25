import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchvision

class CNN(L.LightningModule):
    def __init__(self, model_architecture, loss_function, optimizer, learning_rate, weight_decay):
        super().__init__()
        self.model_architecture = model_architecture
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if model_architecture == 'convnext_small':
            model_echo = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT)
            model_echo.classifier[-1] = torch.nn.Linear(model_echo.classifier[-1].in_features, 1)
        elif model_architecture == 'resnet18':
            model_echo = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            num_features = model_echo.fc.in_features
            model_echo.fc = torch.nn.Linear(num_features, 1)
        elif model_architecture == 'resnet50':
            model_echo = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            num_features = model_echo.fc.in_features
            model_echo.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1)
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_architecture}")

        self.model = model_echo

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if self.loss_function == 'mse':
            loss = F.mse_loss(y_hat.squeeze(), y)
        elif self.loss_function == 'l1':
            loss = F.l1_loss(y_hat.squeeze(), y)
        elif self.loss_function == 'huber':
            loss = F.huber_loss(y_hat.squeeze(), y, delta=1.0)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if self.loss_function == 'mse':
            loss = F.mse_loss(y_hat.squeeze(), y)
        elif self.loss_function == 'l1':
            loss = F.l1_loss(y_hat.squeeze(), y)
        elif self.loss_function == 'huber':
            loss = F.huber_loss(y_hat.squeeze(), y, delta=1.0)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_r2", r2_score(y.detach().cpu(), y_hat.squeeze().detach().cpu()), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def forward(self, x):
        return self.model(x)

    def predict_si(self, x):
        with torch.no_grad():
            si = self.model(x).squeeze()
        return si
