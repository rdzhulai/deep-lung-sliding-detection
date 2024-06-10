import torch
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score, Specificity
import time

def train_loop(model, criterion, optimizer, tqdm_train_dataloader, history, device, lr_scheduler=None, grad_clip=None):
    loss_sum = 0.0
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1 = F1Score(task="binary")
    specificity = Specificity(task="binary")
    confusion_matrix = ConfusionMatrix(task="binary")
    
    start_time = time.time()
    for batch_idx, (batch_frames, batch_labels) in enumerate(tqdm_train_dataloader, 1):
        batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_frames).squeeze(1)
        loss = criterion(output, batch_labels)
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()
        
        loss_sum += loss.item() / batch_frames.size(0)
        
        batch_labels = batch_labels.cpu()
        predicted = torch.sigmoid(output).cpu() > 0.5
        accuracy.update(predicted, batch_labels)
        precision.update(predicted, batch_labels)
        recall.update(predicted, batch_labels)
        f1.update(predicted, batch_labels)
        specificity.update(predicted, batch_labels)
        confusion_matrix.update(predicted, batch_labels)
        
        tqdm_train_dataloader.set_postfix({'train_loss': loss_sum / batch_idx})
    end_time = time.time()
    
    history['time'].append(end_time-start_time)
    history['loss'].append(loss_sum/len(tqdm_train_dataloader))
    history['accuracy'].append(accuracy.compute())
    history['precision'].append(precision.compute())
    history['recall'].append(recall.compute())
    history['f1'].append(f1.compute())
    history['specificity'].append(specificity.compute())
    history['confusion_matrix'].append(confusion_matrix.compute())

        
def val_loop(model, criterion, tqdm_dataloader, history, device):
    model.eval()
    
    loss_sum = 0.0
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1 = F1Score(task="binary")
    specificity = Specificity(task="binary")
    confusion_matrix = ConfusionMatrix(task="binary")
    
    start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (batch_frames, batch_labels) in enumerate(tqdm_dataloader, 1):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            output = model(batch_frames).squeeze(1)
            
            loss_sum += criterion(output, batch_labels).item() / batch_frames.size(0)
            
            batch_labels = batch_labels.cpu()
            predicted = torch.sigmoid(output).cpu() > 0.5
            accuracy.update(predicted, batch_labels)
            precision.update(predicted, batch_labels)
            recall.update(predicted, batch_labels)
            f1.update(predicted, batch_labels)
            specificity.update(predicted, batch_labels)
            confusion_matrix.update(predicted, batch_labels)
            
            tqdm_dataloader.set_postfix({'val_loss': loss_sum / batch_idx})
    end_time = time.time()
    
    history['time'].append(end_time-start_time)
    history['loss'].append(loss_sum/len(tqdm_dataloader))
    history['accuracy'].append(accuracy.compute())
    history['precision'].append(precision.compute())
    history['recall'].append(recall.compute())
    history['f1'].append(f1.compute())
    history['specificity'].append(specificity.compute())
    history['confusion_matrix'].append(confusion_matrix.compute())
    
    model.train()
    
def test_loop(model, criterion, tqdm_dataloader, history, device):
    model.eval()
    
    loss_sum = 0.0
    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1 = F1Score(task="binary")
    specificity = Specificity(task="binary")
    confusion_matrix = ConfusionMatrix(task="binary")
    
    start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (batch_frames, batch_labels) in enumerate(tqdm_dataloader, 1):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            output = model(batch_frames).squeeze(1)
            
            loss_sum += criterion(output, batch_labels).item() / batch_frames.size(0)
            
            batch_labels = batch_labels.cpu()
            predicted = torch.sigmoid(output).cpu() > 0.5
            accuracy.update(predicted, batch_labels)
            precision.update(predicted, batch_labels)
            recall.update(predicted, batch_labels)
            f1.update(predicted, batch_labels)
            specificity.update(predicted, batch_labels)
            confusion_matrix.update(predicted, batch_labels)
            
            tqdm_dataloader.set_postfix({'test_loss': loss_sum / batch_idx})
    end_time = time.time()
    
    history['time'][-1] = end_time-start_time
    history['loss'][-1] = loss_sum/len(tqdm_dataloader)
    history['accuracy'][-1] = accuracy.compute()
    history['precision'][-1] = precision.compute()
    history['recall'][-1] = recall.compute()
    history['f1'][-1] = f1.compute()
    history['specificity'][-1] =  specificity.compute()
    history['confusion_matrix'][-1] = confusion_matrix.compute()
    
    model.train()
