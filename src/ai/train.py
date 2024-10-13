from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


from src.ai.model import CommandIdentifier
from src.ai.dataset import CommandDataset
from src.ai import DEVICE

from src.util import get_labels

from src.config.neural_net import MODEL_FILE_PATH, NN_TRAIN_EPOCHS


def train_model() -> None:
    """
    Train the neural network model for command identification.

    This function will create a model, create a DataLoader from the dataset, train the model using the DataLoader, and then save the model to a file.
    """
    hidden_size = 16
    
    dataset = CommandDataset()

    model = CommandIdentifier(input_size=len(dataset.stemmed_vocabulary), hidden_size=hidden_size, num_classes=len(get_labels()))
    model.to(DEVICE)

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    accuracy_values = []
    loss_values = []
    for epoch in (pbar := tqdm(range(NN_TRAIN_EPOCHS))):
        loss_val = 0
        accuracy_val = 0
        
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item
            
            optimizer.step()
            # print("\n", [get_labels().index(i) for i in outputs.detach().numpy()])
            # print("\n", labels.detach().numpy().shape)
            
            acc_current = f1_score([i.argmax() for i in outputs.detach().numpy()], labels.detach().numpy(), average='macro')
            accuracy_val += acc_current
            
        accuracy_values.append(accuracy_val/len(train_dataloader))
        loss_values.append(loss_val/len(train_dataloader))
            
        pbar.set_description(f"Epoch {epoch+1} >> Average loss: {loss_val/len(train_dataloader):.4f}, Average accuracy: {accuracy_val/len(train_dataloader):.4f}")
            
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    data = {
        "model_state": model.state_dict(),
        "input_size": len(dataset.stemmed_vocabulary),
        "hidden_size": hidden_size,
        "output_size": len(get_labels())
    }
    torch.save(data, MODEL_FILE_PATH)
    logger.success("NN successfully trained")
    