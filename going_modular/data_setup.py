"Contains the functionality for craeting Pytorch dataloaders for image classification data"

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

NUM_WORK=os.cpu_count()

def create_dataloader(train_dir:str,test_dir:str,transform:transforms.Compose,batch_size:int,num_workers:int=NUM_WORK):
  train_data=datasets.ImageFolder(train_dir,transform=transform)
  test_data=datasets.ImageFolder(test_dir,transform=transform)

  class_name=train_data.classes

  train_dataloader=DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  test_dataloader=DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader,test_dataloader,class_name
