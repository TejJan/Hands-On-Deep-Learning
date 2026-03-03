import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

def test_new_class_index(pretrained_model):
    """
    Your model needs to be able to accept the new class index 10. 
    This function lets you test this on the cpu to get a proper stack trace, 
    you do _not_ need to keep it in your final solution.
    """
    new_class_index = 10
    x = torch.randn(32, 1, 28, 28)
    t = torch.randint(0, 1000, (32, 1, 1, 1))
    c = torch.ones(32, dtype=torch.long) * new_class_index
    pretrained_model.to("cpu")(x, t/1000, c)

def finetune_model(pretrained_model, train_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    
    # Extend embedding layer to support class 10
    old_embedding = pretrained_model.class_embedding
    embedding_dim = old_embedding.embedding_dim
    new_embedding = nn.Embedding(11, embedding_dim).to(device)
    
    # Initialize new embedding with existing weights
    with torch.no_grad():
        new_embedding.weight[:10] = old_embedding.weight.data
        # Initialize class 10 as mean of existing classes
        new_embedding.weight[10] = old_embedding.weight.data.mean(dim=0)
    
    pretrained_model.class_embedding = new_embedding
    pretrained_model.n_classes = 11
    
    # Enable all parameters for training
    for param in pretrained_model.parameters():
        param.requires_grad = True
    
    # Training configuration
    epochs = 200
    batch_size = 8
    lr = 0.00005
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(pretrained_model.parameters(), lr=lr, weight_decay=0.01)
    
    best_loss = float('inf')
    best_model = None
    
    pretrained_model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, batch_data in enumerate(train_loader):
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            
            # Sample timestep for diffusion
            t = torch.randint(0, 1001, (images.shape[0], 1, 1, 1)).to(device)
            t = t / 1000.0
            
            # Generate noise
            noise = torch.randn_like(images)
            
            # Create noisy images
            x_t = t * noise + (1 - t) * images
            
            optimizer.zero_grad(set_to_none=True)
            
            # Predict velocity
            v_pred = pretrained_model(x_t, t, labels)
            v_true = noise - images
            
            # Compute loss
            loss = nn.functional.mse_loss(v_pred, v_true)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = deepcopy(pretrained_model)
        
        if (epoch + 1) % 40 == 0:
            print(f'Epoch: {epoch + 1} loss: {avg_loss:.5f}')
    
    # Restore best model
    if best_model is not None:
        pretrained_model = best_model
    
    pretrained_model.eval()
    test_new_class_index(pretrained_model)
    
    return pretrained_model
