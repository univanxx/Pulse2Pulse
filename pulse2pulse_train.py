#===========================================
# Reference: https://github.com/mazzzystar/WaveGAN-pytorch
#===========================================

import argparse
import os
from tqdm import tqdm
import numpy as np
import pickle

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
from torch import autograd

import matplotlib.pyplot as plt

# Model specific
from models.pulse2pulse import WaveGANGenerator as Pulse2PuseGenerator
from models.pulse2pulse import WaveGANDiscriminator as Pulse2PulseDiscriminator

import sys
sys.path.append("../../data")
from data_modules import ECGDataset

# from data.ecg_data_loader import ECGDataSimple as ecg_data
from utils.utils import calc_gradient_penalty, get_plots_RHTM_10s, get_plots_all_RHTM_10s



torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device_id", type=int, default=1, help="Device ID to run the code")
# parser.add_argument("--py_file",default=os.path.abspath(__file__)) # store current python file


#==============================
# Directory and file handling
#==============================
# parser.add_argument("--data_dirs", default=["/home/vajira/DL/Pulse2Pulse/sample_ecg_data", 
#                                             ], help="Data roots", nargs="*")

parser.add_argument("--out_dir", 
                    default="./output",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="./tensorboard",
                    help="Folder to save output of tensorboard")
#======================
# Hyper parameters
#======================
parser.add_argument("--bs", type=int, default=1, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--num_epochs", type=int, default=4000, help="number of epochs of training")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs used in models")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval to plot model results")
parser.add_argument("--save_interval", type=int, default=100, help="Interval to save models")

# Checkpoint path to retrain or test models
parser.add_argument("--checkpoint_path", default="", help="Check point path to retrain or test models")

parser.add_argument('-ms', '--model_size', type=int, default=50,
                        help='Model size parameter used in WaveGAN')
parser.add_argument('--lmbda', default=10.0, help="Gradient penalty regularization factor")

# Action handling 
parser.add_argument("--action", required=True, type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])

parser.add_argument("--data_dir", required=True,
                    help="Folder to the data")

parser.add_argument("--exp_name", type=str, required=True, help="A name to the experiment which is used to save checkpoitns and tensorboard output")


opt = parser.parse_args()
print(opt)

#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/cps")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, opt.exp_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)



#==========================================
# Tensorboard
#==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)


#==========================================
# Prepare Data
#==========================================
def prepare_data():

    with open(os.path.join(opt.data_dir, "label2id.pickle"), "rb") as f:
        label2id = pickle.load(f)
    selected_classes = ['426783006', '39732003', '164873001', '164889003', '427084000', '270492004', '426177001', '164934002']

    dataset = ECGDataset(opt.data_dir, "ptb-xl", label2id, selected_classes, option="train")
    print("Dataset size=", len(dataset))
    
    dataloader = torch.utils.data.DataLoader( dataset,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=0
    )

    return dataloader, len(selected_classes)

#===============================================
# Prepare models
#===============================================
def prepare_model(num_classes):

    netG = Pulse2PuseGenerator(num_classes, model_size=opt.model_size, ngpus=opt.ngpus, upsample=True)
    netD = Pulse2PulseDiscriminator(model_size=opt.model_size, ngpus=opt.ngpus, verbose=False)

    netG = netG.to(device)
    netD = netD.to(device)

    return netG, netD

#====================================
# Run training process
#====================================
def run_train():
    dataloaders, num_classes = prepare_data() 

    netG, netD = prepare_model(num_classes)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    train(netG, netD, optimizerG, optimizerD, dataloaders, num_classes)

def train(netG, netD, optimizerG, optimizerD, dataloader, num_classes):

    for epoch in tqdm(range(opt.start_epoch+1, opt.num_epochs+1)):

        len_dataloader = len(dataloader)
        print("Length of Dataloader:", len_dataloader)

        train_G_flag = False
        D_cost_train_epoch = []
        D_wass_train_epoch = []
        G_cost_epoch = []
    
        for i, sample in tqdm(enumerate(dataloader, 0)):

            image, label = sample
            label = label.to(device)

            if (i+1) % 5 == 0:
                train_G_flag = True

            # Set Discriminator parameters to require gradients.
            #print(train_G_flag)
            for p in netD.parameters():
                p.requires_grad = True

            #one = torch.Tensor([1]).float()
            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1

            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator
            #############################

            real_ecgs = image.to(device)
            #print("real ecgs shape", real_ecgs.shape)
            b_size = real_ecgs.size(0)

            netD.zero_grad()


            # Noise
            noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
            noise = noise.to(device)
            noise_Var = Variable(noise, requires_grad=False)

            # real_data_Var = numpy_to_var(next(train_iter)['X'], cuda)

            # a) compute loss contribution from real training data
            D_real = netD(real_ecgs)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var, label).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_ecgs,
                                                    fake.data, b_size, opt.lmbda,
                                                    use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()


            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)


            #############################
            # (3) Train Generator
            #############################
            if train_G_flag:
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
                
                noise = noise.to(device)
                noise_Var = Variable(noise, requires_grad=False)

                fake = netG(noise_Var, label)
                # print(fake.shape)
                G = netD(fake)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                # Record costs
                #if cuda:
                G_cost_cpu = G_cost.data.cpu()
                #print("g_cost=",G_cost_cpu)
                G_cost_epoch.append(G_cost_cpu)
                #print("Epoch{} - {}_G_cost_cpu:{}".format(epoch, i, G_cost_cpu))
                #G_cost_epoch.append(G_cost_cpu.data.numpy())
                train_G_flag =False

                #print("real ecg:", real_ecgs.shape)
                #print("fake ecg:", fake.shape)
            # if i == 0: # take the first batch to plot
            #     real_ecgs_to_plot = real_ecgs
            #     fake_to_plot = fake
            #    break
        #print(G_cost_epoch)

        D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
        D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
        G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

        
        writer.add_scalar("D_cost_train_epoch_avg",D_cost_train_epoch_avg ,epoch)
        writer.add_scalar("D_wass_train_epoch_avg",D_wass_train_epoch_avg ,epoch)
        writer.add_scalar("G_cost_epoch_avg ",G_cost_epoch_avg  ,epoch)

        print("Epochs:{}\t\tD_cost:{}\t\t D_wass:{}\t\tG_cost:{}".format(
                    epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))

         
        if epoch % opt.checkpoint_interval == 0:
            
            lab_1 = torch.zeros(num_classes).to(device)
            lab_2 = torch.zeros(num_classes).to(device)
            lab_2[-1] = 1

            for label_i in [lab_1, lab_2]:
                fig, axs = plt.subplots(3, 2, figsize=(20, 12))
                noise = torch.Tensor(2, 8, 5000).uniform_(-1, 1).to(device)
                with torch.no_grad():
                    fake = netG(noise, label_i[None,:].to(device)).cpu().numpy()
                    # import pdb
                    # pdb.set_trace()
                    fig.suptitle("Generated I, III and VI leads for class {}".format(label_i))
                    for i in range(2):
                        axs[0][i].plot(fake[i][0])
                        axs[1][i].plot(fake[i][1])
                        axs[2][i].plot(fake[i][2])
                # plt.savefig("test.png")
                writer.add_figure('label_{}/generated ecgs'.format(label_i), fig, epoch)
                    
            # fig = get_plots_RHTM_10s(real_ecgs_to_plot[0].detach().cpu(), fake_to_plot[0].detach().cpu())
            # fig_2 = get_plots_all_RHTM_10s(real_ecgs_to_plot.detach().cpu(), fake_to_plot.detach().cpu())
            # writer.add_figure("sample", fig, epoch)
            # writer.add_figure("sample_batch", fig_2, epoch)
            
        # Save model    
        if epoch % opt.save_interval == 0:
            save_model(netG, netD, optimizerG, optimizerD, epoch)


#=====================================
# Save models
#=====================================
def save_model(netG, netD, optimizerG, optimizerD,  epoch):
   
    check_point_name = "gan_p2p" + "_epoch:{}.pt".format(epoch) # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
        # "train_loss": train_loss,
        #"val_loss": validation_loss
    }, check_point_path)

#====================================
# Re-train process
#====================================
def run_retrain():
    print("run retrain started........................")
    netG, netD = prepare_model()

    #netG.cpu()
    #netD.cpu()

    # loading checkpoing
    chkpnt = torch.load(opt.checkpoint_path, map_location="cpu")

    netG.load_state_dict(chkpnt["netG_state_dict"])
    netD.load_state_dict(chkpnt["netD_state_dict"])

    netG = netG.to(device)
    netD = netD.to(device)

    print("model loaded from checkpoint=", opt.checkpoint_path)

    # setup start epoch to checkpoint epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders)
    

#=====================================
# Check model
#====================================
def check_model_graph():
    netG, netD = prepare_model()
    print(netG)
    netG = netG.to(device)
    netD = netD.to(device)

    summary(netG, (8,5000))
    summary(netD, (8, 5000))




if __name__ == "__main__":

    data_loaders, _ = prepare_data()
    print(vars(opt))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is started..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is started..!")
        run_retrain()
        pass
    elif opt.action == "inference":
        print("Inference process is started..!")
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")

    # Finish tensorboard writer
    writer.close()
