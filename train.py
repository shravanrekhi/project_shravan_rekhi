# train.py
import torch

def train_pneumonia_model(model, num_epochs, train_loader, loss_fn, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0
        
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * batch.size(0)
            
        avg_loss = training_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")