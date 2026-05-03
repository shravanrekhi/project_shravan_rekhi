# predict.py
import torch
from dataset import PneumoniaImgDataset, get_dataloader
from model import PneumoniaCNN
import config

def classify_xrays(list_of_img_paths, weights_path="_checkpoints/_final_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = PneumoniaCNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create dummy labels because the dataset class requires them
    dummy_labels = [0] * len(list_of_img_paths)
    infer_dataset = PneumoniaImgDataset(list_of_img_paths, dummy_labels, is_train=False)
    infer_loader = get_dataloader(infer_dataset, is_train=False)

    all_labels = []
    
    with torch.no_grad():
        for batch, _ in infer_loader:
            batch = batch.to(device)
            logits = model(batch)
            
            # Using the threshold defined in config.py
            for l in logits:
                if l.item() > config.threshold:
                    all_labels.append("Pneumonia")
                else:
                    all_labels.append("Normal")
                    
    return all_labels