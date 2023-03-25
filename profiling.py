# https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import shutil
import os

from networks import *

# 1) Configurations
checkpoint_dir = "checkpoints/"
checkpoint_name = "temporary.pt"

# 2) Hyper parameters and some other parameters
hyper_parameters = {'train_batch_size':128,
                    'validation_batch_size':128,
                    }

parameters = {'epoch':1,
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
optimizer = optim.Adam(model.parameters())
evaluator = Evaluator()
train_writer = SummaryWriter(f"tensorboard/{checkpoint_name[:-3]}/train/")
validation_writer = SummaryWriter(f"tensorboard/{checkpoint_name[:-3]}/validation/")

# 5) Computational graph
writer = SummaryWriter("tensorboard/profiler/")
writer.add_graph(model, torch.rand((128, 1, 28, 28), device=model.device))
writer.close()

# 6) Memory and Time
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('tensorboard/profiler/'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
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
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            # write
            train_writer.add_scalar("eval/cost", round(cost.item(), 5), global_step_counter)
            train_writer.add_scalar("eval/accuracy", round(accuracy, 5), global_step_counter)
            global_step_counter += 1
            tqdm_batch_bar.update(1)
        prof.step()
        
        # validation
        if epoch_i % parameters['check_period'] == 0:
            model.eval()
            with torch.no_grad():
                total_cost = 0
                total_accuracy = 0
                tqdm_batch_bar = tqdm(total=len(validation_dataloader), desc="Validation Mini-Batch Counter", unit='iter', leave=False)
                for data, label in validation_dataloader:
                    # data
                    data, label = data.to(model.device), torch.nn.functional.one_hot(label, num_classes=10).to(model.device)
                    
                    # run
                    capsule_predictions, reconstructions = model(data)
                    cost = model.cost(data, label, reconstructions, capsule_predictions).item()
                    accuracy = evaluator.accuracy(capsule_predictions.detach(), label.detach())
                    
                    total_cost += cost * (data.shape[0]/len(validation_dataloader.dataset))
                    total_accuracy += accuracy * (data.shape[0]/len(validation_dataloader.dataset))

                    tqdm_batch_bar.update(1)
                    
                    
                if total_accuracy > best_accuracy:
                    best_accuracy = total_accuracy
                    torch.save(model.state_dict(), checkpoint_dir+checkpoint_name)
            
            # write
            validation_writer.add_scalar("eval/cost", round(total_cost, 5), global_step_counter)
            validation_writer.add_scalar("eval/accuracy", round(total_accuracy, 5), global_step_counter)
            model.train()
            prof.step()
            
        tqdm_epoch_bar.update(1)

# 7) Delete temporary files
os.remove(checkpoint_dir+checkpoint_name)
shutil.rmtree(f"tensorboard/{checkpoint_name[:-3]}", ignore_errors=False, onerror=None)