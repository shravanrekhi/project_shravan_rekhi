# interface.py

from model import PneumoniaCNN as TheModel
from train import train_pneumonia_model as the_trainer
from predict import classify_xrays as the_predictor
from dataset import PneumoniaImgDataset as TheDataset
from dataset import get_dataloader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs