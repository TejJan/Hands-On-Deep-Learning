import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Do not change function signature
def init_model() -> nn.Module:
  # Your code here
  
  class MNISTNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.layers = nn.ModuleList([
        nn.Linear(784, 196),  # 256 neurons taking 28 * 28 = 784 inputs
        nn.ReLU(),            # non linearity
        nn.Linear(196, 10)    # output layer, 
      ])
    
    # Input dimension: [B, 1, 28, 28]  (B = batch size, grayscale MNIST image)
    # Output dimension: [B, 10]        (digits 0–9)
    
    # Hint: you can flatten the image in your forward loop
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = x.flatten(1, 3)   # [B, 1, 28, 28] -> [B, 784] 
                            # by flattening grayscale, width and height dimensions
      for layer in self.layers:
        x = layer(x)

      return x
  
  model = MNISTNet()
  return model
    

# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
  # Your code
  batch_size = 96
  learning_rate = 0.005
  verbosity = 3
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # optimizer 
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = nn.CrossEntropyLoss()
  
  # dataloader for batch processing
  dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
  size = len(dataloader.dataset)
  acc_loss = 0
  acc_count = 0
  last_print_point = 0
  current = 0
  model.train()
  
  for batch, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    
    # forward pass
    pred = model(X)
    
    # loss calculation
    loss_val = loss_fn(pred, y)
    acc_loss += loss_val.item()
    acc_count += 1
    
    # zero the gradients computed in the previous step
    optimizer.zero_grad()

    # calculate the gradients of the parameters of the net
    loss_val.backward()

    # use the gradients to update the weights of the network
    optimizer.step()

    # compute how many datapoints have already been used for training
    current = batch * len(X)

    # report on the training progress roughly every 10% of the progress
    if verbosity >= 3 and (current - last_print_point) / size >= 0.1:
        loss_val = loss_val.item()
        last_print_point = current
        # print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    
        print(f"lr: {learning_rate:>7f} batch_size: {batch_size:>4f} layers: {len(model.layers):>3f}")  
        print(acc_loss / acc_count)
  # Uncomment to modify the dataset (optional)
  # class MyDataset(Dataset):
  #     def __init__(self, base_dataset: Dataset):
  #         self.base_dataset = base_dataset
  #
  #     def __len__(self) -> int:
  #         return len(self.base_dataset)
  #
  #     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
  #         data, target = self.base_dataset[idx]
  #         return data, target
  #
  # train_dataset = MyDataset(dev_dataset)

  return model