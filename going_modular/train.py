
import os
import torch
from torchvision import transforms
import data_setup,engine,model_builder,utils

NUM_EPOCHS=10
BATCH_SIZE=32
HIDDEN_UNITS=10
LRATE=0.01

train_dir='data/pizza_steak_sushi_mod/train'
test_dir='data/pizza_steak_sushi_mod/test'

device= 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform=transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

train_dataloader,test_dataloader,class_names=data_setup.create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

model=model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)).to(device)

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=LRATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir='models',
                 model_name='script_based_model.pth')
