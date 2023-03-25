import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from networks import *

# 1) Configuration
checkpoint_dir = "checkpoints/"
try:
    checkpoints = os.listdir(checkpoint_dir)
except FileNotFoundError:
    os.mkdir(checkpoint_dir)
    checkpoints = os.listdir(checkpoint_dir)
if len(checkpoints):
    last_checkpoint_index = max([int(cp.split("_")[1][:-3]) for cp in checkpoints])
else:
    last_checkpoint_index = 0
checkpoint_name = f"checkpoint_{last_checkpoint_index+1}.pt"

# 2) Hyper parameters and some other parameters
hyper_parameters = {'train_batch_size':128,
                    'validation_batch_size':128,
                    }

parameters = {'start_over':True, # whether to start training from beginning or keep training from the last checkpoint
              'epoch':20,
              'check_period':1, # the check period. validation cost will be checked after every check_period epoch.
              'continue_from':None} # which checkpoint to use. you can ignore if start_over=True. you can give 
# continue_from=None and start_over=False to continue from the last checkpoint. 

# 3) Data
train_dataset = torchvision.datasets.MNIST("datasets/", train=True, download=True,
                                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                            )
validation_dataset = torchvision.datasets.MNIST("datasets/", train=False, download=True,
                                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                                )
train_dataloader = DataLoader(train_dataset, hyper_parameters['train_batch_size'], shuffle=True)
validation_dataloader = DataLoader(validation_dataset, hyper_parameters['validation_batch_size'], shuffle=True)

# 4) Models
model = CapsuleNetwork(device="cuda:0" if torch.cuda.is_available() else "cpu")
#torch.compile(model)
if parameters['start_over']:
    pass
else:
    model.load_state_dict(torch.load(checkpoint_dir + f"checkpoint_{last_checkpoint_index}.pt"))
optimizer = optim.Adam(model.parameters())
evaluator = Evaluator()
train_writer = SummaryWriter(f"tensorboard/{checkpoint_name[:-3]}/train/")
validation_writer = SummaryWriter(f"tensorboard/{checkpoint_name[:-3]}/validation/")

# 5) Run
best_accuracy = -1
global_step_counter = 0
tqdm_epoch_bar = tqdm(total=parameters['epoch'], desc="Epoch Counter", unit='iter', leave=True)
for epoch_i in range(parameters['epoch']):
    
    # train
    tqdm_batch_bar = tqdm(total=len(train_dataloader), desc="Training Mini-Batch Counter", unit='iter', leave=False)
    for data, label in train_dataloader:
        # data
        data, label = data.to(model.device), torch.nn.functional.one_hot(label, num_classes=10).to(model.device)
        
        # run
        capsule_predictions, reconstructions = model(data, label)
        cost = model.cost(data, label, reconstructions, capsule_predictions)
        accuracy = evaluator.accuracy(capsule_predictions.detach(), label.detach())
        one_digit_accuracy = evaluator.one_digit_accuracy(capsule_predictions.detach(), label.detach())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # write
        train_writer.add_scalar("eval/cost", round(cost.item(), 5), global_step_counter)
        train_writer.add_scalar("eval/accuracy", round(accuracy, 5), global_step_counter)
        train_writer.add_scalar("eval/one_digit_accuracy", round(one_digit_accuracy, 5), global_step_counter)
        global_step_counter += 1
        tqdm_batch_bar.update(1)
    
    # validation
    if epoch_i % parameters['check_period'] == 0:
        model.eval()
        with torch.no_grad():
            total_cost = 0
            total_accuracy = 0
            total_one_digit_accuracy = 0
            tqdm_batch_bar = tqdm(total=len(validation_dataloader), desc="Validation Mini-Batch Counter", unit='iter', leave=False)
            for data, label in validation_dataloader:
                # data
                data, label = data.to(model.device), torch.nn.functional.one_hot(label, num_classes=10).to(model.device)
                
                # run
                capsule_predictions, reconstructions = model(data)
                cost = model.cost(data, label, reconstructions, capsule_predictions).item()
                accuracy = evaluator.accuracy(capsule_predictions.detach(), label.detach())
                one_digit_accuracy = evaluator.one_digit_accuracy(capsule_predictions.detach(), label.detach())
                
                total_cost += cost * (data.shape[0]/len(validation_dataloader.dataset))
                total_accuracy += accuracy * (data.shape[0]/len(validation_dataloader.dataset))
                total_one_digit_accuracy += one_digit_accuracy * (data.shape[0]/len(validation_dataloader.dataset))

                tqdm_batch_bar.update(1)
                
            if total_accuracy > best_accuracy:
                best_accuracy = total_accuracy
                torch.save(model.state_dict(), checkpoint_dir+checkpoint_name)
            
        # write
        validation_writer.add_scalar("eval/cost", round(total_cost, 5), global_step_counter)
        validation_writer.add_scalar("eval/accuracy", round(total_accuracy, 5), global_step_counter)
        validation_writer.add_scalar("eval/one_digit_accuracy", round(total_one_digit_accuracy, 5), global_step_counter)
        model.train()
        
        
    tqdm_epoch_bar.update(1)