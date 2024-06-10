import torch

def get_batch_size(
    model,
    criterion,
    optimizer,
    input_shape,
    output_shape,
    dataset_size,
    device,
    max_batch_size=None,
    num_iterations=5,
):
    model.to(device)
    model.train(True)
    
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size > max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = dataset_size // 2
            break
        
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand((batch_size, *input_shape), device=device)
                targets = torch.rand((batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = criterion(outputs, targets) 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    
    return batch_size
