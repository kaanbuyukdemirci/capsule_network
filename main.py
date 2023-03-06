import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import itertools
from tqdm import tqdm

from networks import *

# I recommend reading the script first (follow 1, 2, 3)
def execute(hyperparameters:dict, train_dataloader:DataLoader, test_dataloader:DataLoader, 
            model:CapsuleNetwork, optimizer:torch.optim.Adam, evaluator:Evaluator, epoch:int,
            check_period:int):
    global_step_counter = 0
    tqdm_epoch_bar = tqdm(total=epoch, desc="Epoch Counter", unit='iter', leave=True)
    for epoch_i in range(epoch):
        tqdm_batch_bar = tqdm(total=len(train_dataloader), desc="Mini-Batch Counter", unit='iter', leave=False)
        for batch_i, (data, label) in enumerate(train_dataloader):
            # load the data
            data, label = data.to(model.device), torch.nn.functional.one_hot(label, num_classes=10).to(model.device)
            
            # training:
            # get the capsule predictions and reconstructions:
            capsule_predictions, reconstructions = model(data, label)
            # calculate lost and accuracy:
            cost = model.cost(data, label, reconstructions, capsule_predictions)
            # zero grad, backward and step
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
             # loss/accuracy
            if batch_i % check_period == 0:
                model.eval()
                with torch.no_grad():
                    # 1) train dataset (we need to do the forward pass again due to masking mechanism):
                    # data:
                    # (just samples a batch from the dataset, I couldn't find a good way of doing that. so don't worry about the following 3 lines)
                    indexes = torch.randint(0, len(train_dataloader.dataset), (hyperparameters['batch_size_test'],))
                    data = torch.cat([train_dataloader.dataset[index][0].view(1, *train_dataloader.dataset[index][0].shape) for index in indexes], dim=0).to(model.device)
                    label = torch.nn.functional.one_hot(torch.tensor([train_dataloader.dataset[index][1] for index in indexes]), num_classes=10).to(model.device)
                    # running the model:
                    capsule_predictions, reconstructions = model(data)
                    # cost:
                    cost = round(model.cost(data, label, reconstructions, capsule_predictions).item(), 2)
                    writer.add_scalar("Train Cost", cost, global_step_counter)
                    # accuracy: 
                    accuracy = round(evaluator.accuracy(capsule_predictions.detach(), label.detach()), 2)
                    writer.add_scalar("Train Accuracy", accuracy, global_step_counter)
                    
                    # 1) test dataset:
                    # data:
                    indexes = torch.randint(0, len(test_dataloader.dataset), (hyperparameters['batch_size_test'],))
                    data = torch.cat([test_dataloader.dataset[index][0].view(1, *test_dataloader.dataset[index][0].shape) for index in indexes], dim=0).to(model.device)
                    label = torch.nn.functional.one_hot(torch.tensor([test_dataloader.dataset[index][1] for index in indexes]), num_classes=10).to(model.device)
                    # running the model:
                    capsule_predictions, reconstructions = model(data)
                    # cost:
                    cost = round(model.cost(data, label, reconstructions, capsule_predictions).item(), 2)
                    writer.add_scalar("Test Cost", cost, global_step_counter)
                    # accuracy: 
                    accuracy = round(evaluator.accuracy(capsule_predictions.detach(), label.detach()), 2)
                    writer.add_scalar("Test Accuracy", accuracy, global_step_counter)
                    
                    global_step_counter += 1
                model.train()
            tqdm_batch_bar.update(1)
        tqdm_epoch_bar.update(1)

# 1) set the configuration
start_over = True # keep training from model_load, or start over. Saves no matter which.
model_save_dir = "weight_saves/"
model_save_name = "saved_model.pt"
model_load_dir = "weight_saves/"
model_load_name = "saved_model.pt"

# 2) set the hyper-parameters and some other stuff
# add any hyperparameter that you want to use in 'execute' function to this dictionary
hyperparameters = {"batch_size_train":[128], # while training.
                   "batch_size_test":[100]} # while testing.
unique_writer_id = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') # e.g: 2023_03_05_21_00_09, for tensorboard
epoch = 100
check_period = 10 # how often (mini-batch-wise) the training/test loss/accuracy will be calculated 

# 3) get the inputs for the 'execute' function above ready and pass them.
# the following 3 lines just samples from the hyperparameter space a set of hyperparameters,
# and hold it as a dictionary for the duration of the iteration.
# it does so that you can add hyperparameters more easily.
values_list = [hyperparameters[key] for key in hyperparameters.keys()]
for values in itertools.product(*values_list):
    hyperparameters = dict(zip(hyperparameters.keys(), values))
    
    # 3.1) set the data ready
    train_dataset = torchvision.datasets.MNIST("datasets/", train=True, download=True,
                                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                               )
    test_dataset = torchvision.datasets.MNIST("datasets/", train=False, download=True,
                                              transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                              )
    train_dataloader = DataLoader(train_dataset, hyperparameters['batch_size_train'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, hyperparameters['batch_size_test'], shuffle=True)
    
    # 3.2) set the model, optimizer, evaluator, tensorboard and stuff
    model = CapsuleNetwork(device="cuda:0" if torch.cuda.is_available() else "cpu")
    if start_over:
        pass
    else:
        model.load_state_dict(torch.load(model_load_dir+model_load_name))
    optimizer = optim.Adam(model.parameters())
    evaluator = Evaluator()
    # just some formatting so it look better:
    name = values.__str__()[1:-1].replace(' ', '').replace(',', '_').replace("'", '').replace(':', '_')
    writer = SummaryWriter(f"tensorboard/MNIST/{unique_writer_id}/{name}")

    # 3.3) run
    execute(hyperparameters, train_dataloader, test_dataloader, model, optimizer, evaluator, epoch, check_period)

torch.save(model.state_dict(), model_save_dir+model_save_name)


# TODO: profile, memory and runtime.
# TODO: plot capsule magnitudes
# TODO: plot gradient magnitudes
# TODO: capsule layer v2