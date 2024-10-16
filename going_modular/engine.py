
from typing import Tuple,Dict,List
import torch
from tqdm.auto import tqdm

def train_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,device:torch.device)->Tuple[float,float]:

  model.train()

  train_loss,train_acc=0,0

  for batch,(x,y) in enumerate(dataloader):
    #print(x.dtype)
    x,y=x.to(device),y.to(device)

    y_pred=model(x)

    loss=loss_fn(y_pred,y)
    train_loss+=loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
    train_acc+=(y_pred_class==y).sum().item()/len(y_pred)

  train_loss=train_loss/len(dataloader)
  train_acc=train_acc/len(dataloader)
  return train_loss,train_acc

def test_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,device:torch.device)->Tuple[float,float]:
  model.eval()

  test_loss,test_acc=0,0
  with torch.inference_mode():
    for batch,(x,y) in enumerate(dataloader):
      x,y=x.to(device),y.to(device)

      test_pred_logits=model(x)

      loss=loss_fn(test_pred_logits,y)
      test_loss+=loss.item()

      test_pred_label=test_pred_logits.argmax(dim=1)
      test_acc+=((test_pred_label==y).sum().item()/len(test_pred_label))

  test_loss=test_loss/len(dataloader)
  test_acc=test_acc/len(dataloader)
  return test_loss,test_acc

def train(model:torch.nn.Module,train_dataloader:torch.utils.data.DataLoader,test_dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,epochs:int,device:torch.device)->Dict[str,List]:
  results={
      'train_loss':[],
      'train_acc':[],
      'test_loss':[],
      'test_acc':[]
  }

  for epoch in tqdm(range(epochs)):
    train_loss,train_acc=train_step(model,train_dataloader,loss_fn,optimizer,device)
    test_loss,test_acc=test_step(model,test_dataloader,loss_fn,device)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} |"
        f'train_acc: {train_acc:.2f} | '
        f'test_loss: {test_loss:.4f} | '
        f'test_acc: {test_acc:.2f}'
    )

    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)
  return results
