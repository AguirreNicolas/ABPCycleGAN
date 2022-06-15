import random, os, math, gc, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path

import CycleGAN_helpers


##################################
### Buffer
##################################
"""
Based on: https://nn.labml.ai/gan/cycle_gan/index.html#section-44
"""
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


##################################
### GP Functions
##################################
"""
Based on: https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html
"""
def gradient_penalty(critic,real,fake,device='cuda'):
  BATCH_SIZE,C,LEN = real.shape
  epsilon = torch.rand(BATCH_SIZE,1,1).repeat(1,C,LEN).to(device)
  interpolated_signal = real*epsilon + fake*(1-epsilon)

  #critic_scores
  mixed_scores = critic(interpolated_signal)
  gradient = torch.autograd.grad(
      inputs = interpolated_signal,
      outputs = mixed_scores,
      grad_outputs = torch.ones_like(mixed_scores),
      create_graph = True,
      retain_graph = True
  )[0]

  gradient = gradient.view(gradient.shape[0],-1)
  gradient_norm = gradient.norm(2,dim=1)
  gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

  return gradient_penalty

##################################
### Generator
##################################
### GRU & LSTM
class Generator_RNN(nn.Module):
    def __init__(self, args,domain):
        super(Generator_RNN, self).__init__()
        self.seq_len = args['LEN_SAMPLES']
        self.batch_size = args['batch_size']
        self.layers_gen  = args['layers_gen_'+domain]
        self.in_features = self.layers_gen['in_features']
        self.out_features = self.layers_gen['out_features']
        self.hidden_dim = self.layers_gen['hidden_dim']
        self.num_layers = self.layers_gen['num_layers']
        self.tanh_output = self.layers_gen['tanh_output']
        self.rnn = args['RNN_GEN']
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional == True else 1
        self.device = args['device']
        ###############################
        ## RNN Layer
        ###############################
        if self.rnn == 'lstm':
          self.first = nn.LSTM(input_size=self.in_features, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, batch_first=True,
                              bidirectional= self.bidirectional
                              )
        if self.rnn == 'gru':
          self.first = nn.GRU(input_size=self.in_features, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, batch_first=True,
                              bidirectional= self.bidirectional
                              )
        ###############################
        ## Output Layer
        ###############################          
        if self.tanh_output == True:
            self.out = nn.Sequential(nn.Linear(self.hidden_dim*self.num_directions,
             self.out_features), nn.Tanh())
        else:
            self.out = nn.Linear(self.hidden_dim*self.num_directions, self.out_features)
        self.init_rnn(self.first)

    ###############################
    ## Init methods
    ###############################
    def init_rnn(self,cell, gain=1):
      # orthogonal initialization of recurrent weights
      for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            nn.init.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)
  
    def init_hidden(self,batch_size):
      h_0 = torch.zeros((self.num_directions*self.num_layers,
                         batch_size,self.hidden_dim),device=self.device)
      if self.rnn == 'lstm':
        c_0  = torch.zeros((self.num_directions*self.num_layers,
                            batch_size,self.hidden_dim),device=self.device)
        return (h_0,c_0)
      if self.rnn == 'gru':
        return (h_0)    

    def forward(self, input, hidden):
      input = input.permute(0,2,1)
      rnn_out, hidden = self.first(input, hidden)
      lin_out = self.out(rnn_out).permute(0,2,1)
      return lin_out


##################################
### Critic/Disc.
##################################

### DCNN
class Discriminator_DCGAN(nn.Module):
    def __init__(self,args,domain):
        super(Discriminator_DCGAN, self).__init__()
        self.layers_critic = args['layers_critic_'+ domain]
        self.in_channels = self.layers_critic[1]['f_in'] 
        self.in_lenght = self.layers_critic[1]['input']
        self.gan_mode = args['gan_mode']
        ###############################
        #First
        ###############################
        self.first = nn.Sequential(
            nn.Conv1d(self.layers_critic[1]['f_in'], self.layers_critic[1]['f_out'],
                      self.layers_critic[1]['k'], self.layers_critic[1]['s'], self.layers_critic[1]['p']),
            nn.LeakyReLU(0.2))
            #nn.Mish())
        ###############################
        #Middle Layers
        ###############################
        layers = []
        for i in sorted(self.layers_critic.keys(), reverse=False)[1:]:
          params = self.layers_critic[i]
          if i < len(self.layers_critic.keys()):
            l_i = self._block(params['f_in'],params['f_out'],
                              params['k'],params['s'],params['p'])
            layers.append(l_i)
          ###############################
          # Last Layer
          ###############################          
          if i == len(self.layers_critic.keys()): 
            last_layer = []
            last_layer.append(nn.Conv1d(params['f_in'],params['f_out'],
                              params['k'],params['s'],params['p']))
            if self.gan_mode == 'Vanilla':
              last_layer.append(nn.Sigmoid())
            self.last = nn.Sequential(*last_layer)       
        self.middle = nn.Sequential(*layers)
        self.init_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
      layers = []
      layers.append(nn.Conv1d(in_channels,out_channels,kernel_size,stride,
                      padding,bias=False))
      if self.gan_mode =='WGAN':
        layers.append(nn.InstanceNorm1d(out_channels,affine=True))
      else:
        layers.append(nn.BatchNorm1d(out_channels))
      layers.append(nn.LeakyReLU(0.2))
      return nn.Sequential(*layers)

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d,nn.InstanceNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
      x = self.first(x)
      x = self.middle(x)
      x = self.last(x)
      return x


##################################
### Cycle GAN
##################################

class Cycle_GAN(nn.Module):
    def __init__(self,args):
      super(Cycle_GAN, self).__init__()      
      self.args = args
      self.dict_x = args['dict_x']
      self.dict_y = args['dict_y']
      self.lr = args['lr']
      self.batch_size = args['batch_size']
      self.LEN_SAMPLES = args['LEN_SAMPLES']
      self.RNN_GEN = args['RNN_GEN']
      self.F_GEN = args['F_GEN']
      self.F_CRITIC = args['F_CRITIC']
      self.CRITIC_ITERATIONS = args['CRITIC_ITERATIONS']
      self.GEN_ITERATIONS = args['GEN_ITERATIONS']
      self.LAMBDA_GP_x = args['LAMBDA_GP_x']
      self.LAMBDA_GP_y = args['LAMBDA_GP_y']
      self.LAMBDA_CYCLE_x = args['LAMBDA_CYCLE_x']
      self.LAMBDA_CYCLE_y = args['LAMBDA_CYCLE_y']
      self.LAMBDA_IDENTITY_x = args['LAMBDA_IDENTITY_x']
      self.LAMBDA_IDENTITY_y = args['LAMBDA_IDENTITY_y']
      self.CHANNELS_SIGNAL_x = args['CHANNELS_SIGNAL_x']
      self.CHANNELS_SIGNAL_y = args['CHANNELS_SIGNAL_y']
      self.gan_mode = args['gan_mode']
      self.current_name = args['current_name']      
      self.device = args['device']

      self.gen_xy = Generator_RNN(args,'xy').to(self.device)
      self.critic_y = Discriminator_DCGAN(args,'y').to(self.device)

      self.gen_yx = Generator_RNN(args,'yx').to(self.device)
      self.critic_x = Discriminator_DCGAN(args,'x').to(self.device)

      self.opt_critic_y = optim.Adam(self.critic_y.parameters(), lr=self.lr,betas=(0.5,0.999))
      self.opt_critic_x = optim.Adam(self.critic_x.parameters(), lr=self.lr,betas=(0.5,0.999))

      self.opt_gen_xy = optim.Adam(self.gen_xy.parameters(), lr=self.lr,betas=(0.5,0.999))
      self.opt_gen_yx = optim.Adam(self.gen_yx.parameters(), lr=self.lr,betas=(0.5,0.999))

      # Losses
      self.loss_cycle = nn.L1Loss().to(self.device)
      self.loss_identity = nn.L1Loss().to(self.device)
      if self.gan_mode == 'LSGAN':
        self.loss_gan = nn.MSELoss().to(self.device)
      if self.gan_mode == 'Vanilla':
        print('Lacking Vanilla Loss')

      self.epoch_i = 0
      self.gen_iterations = 0

      self.xy_elements = self.gen_xy,self.critic_y,self.opt_gen_xy,self.opt_critic_y
      self.yx_elements = self.gen_yx,self.critic_x,self.opt_gen_yx,self.opt_critic_x

      self.gen_xy_buffer = ReplayBuffer(max_size=self.batch_size*self.CRITIC_ITERATIONS)
      self.gen_yx_buffer = ReplayBuffer(max_size=self.batch_size*self.CRITIC_ITERATIONS)      

      # Helper Functions (from CycleGAN_helpers)
      self.f_norm_grad_model = CycleGAN_helpers.f_norm_grad_model
      self.func_plot_TB_comparison = CycleGAN_helpers.func_plot_TB_comparison
      self.f_hist_weight_model = CycleGAN_helpers.f_hist_weight_model
      self.gradient_penalty = gradient_penalty
      self.epoch_time = CycleGAN_helpers.epoch_time

    def fit(self,epoch_f,loader,epoch_save,epoch_tb_print,writer,path_save):
      print('-----------Start of Training------------')
      print(f'{self.current_name}')

      # Init losses
      # Identity losses
      LossId_x = torch.tensor(0).to(self.device)
      LossId_y = torch.tensor(0).to(self.device)
      # WGAN + GP Losses
      LossW_x,LossW_y = torch.tensor(0).to(self.device),torch.tensor(0).to(self.device) 
      GP_x,GP_y = torch.tensor(0).to(self.device),torch.tensor(0).to(self.device)

      for epoch in range(self.epoch_i+1,self.epoch_i+epoch_f+1):
        start_time = time.time()
        data_iter = iter(loader)
        i = 0
        while i < len(loader):
          ############################
          # (1) Update G networks
          ###########################
          for p in self.critic_x.parameters():
            p.requires_grad = False # to avoid computation
          for p in self.critic_y.parameters():
            p.requires_grad = False # to avoid computation

          j=0
          while j < self.GEN_ITERATIONS and i < len(loader):
            j += 1
            self.gen_xy.train()
            self.gen_yx.train()
            (x,y) = data_iter.next()
            i += 1
            batch_size_i = x.size()[0]       
            gen_xy_h_0 = self.gen_xy.init_hidden(batch_size_i)
            gen_yx_h_0 = self.gen_yx.init_hidden(batch_size_i)
            # -----------------------
            #  Identity Loss
            # -----------------------
            if  self.LAMBDA_IDENTITY_x > 0 and self.LAMBDA_IDENTITY_y > 0:
              id_x = self.gen_yx(x,gen_yx_h_0)
              id_y = self.gen_xy(y,gen_xy_h_0)
              LossId_x = self.loss_identity(id_x,x) * self.LAMBDA_IDENTITY_x
              LossId_y = self.loss_identity(id_y,y) * self.LAMBDA_IDENTITY_y
            LossId = LossId_x + LossId_y
            # -----------------------
            #  GAN Loss
            # -----------------------
            fake_x = self.gen_yx(y,gen_yx_h_0)
            fake_y = self.gen_xy(x,gen_xy_h_0)
            critic_x_fake = self.critic_x(fake_x)
            critic_y_fake = self.critic_y(fake_y)
            # WGAN
            if self.gan_mode == 'WGAN':
              LossGAN_x = -torch.mean(critic_x_fake)
              LossGAN_y = -torch.mean(critic_y_fake)
            # LSGAN
            if self.gan_mode == 'LSGAN':
              LossGAN_x = 0.5*self.loss_gan(critic_x_fake,torch.ones_like(critic_x_fake))
              LossGAN_y = 0.5*self.loss_gan(critic_y_fake,torch.ones_like(critic_y_fake))
            #Vanilla GAN
            #TODO
            LossGAN = LossGAN_x + LossGAN_y

            # -----------------------
            #  Cycle Loss
            # -----------------------
            cycle_x = self.gen_yx(fake_y,gen_yx_h_0)
            cycle_y = self.gen_xy(fake_x,gen_xy_h_0)
            LossCycle_x = self.loss_cycle(cycle_x,x) * self.LAMBDA_CYCLE_x
            LossCycle_y = self.loss_cycle(cycle_y,y) * self.LAMBDA_CYCLE_y
            LossCycle = LossCycle_x + LossCycle_y

            LossG_x = LossId_x + LossGAN_x + LossCycle_x
            LossG_y = LossId_y + LossGAN_y + LossCycle_y
            LossG = LossG_x + LossG_y

            self.gen_yx.zero_grad()
            self.gen_xy.zero_grad()
            LossG.backward()
            ##############
            # Gradients
            ##############
            #if gen_iterations % epoch_tb_print == 0:
            #gradients_xy = self.f_norm_grad_model(self.gen_xy)
            #gradients_yx = self.f_norm_grad_model(self.gen_yx)
            #writer.add_scalars('Grad_Generator/XY/',gradients_xy, self.gen_iterations)
            #writer.add_scalars('Grad_Generator/YX/',gradients_yx, self.gen_iterations)
        
            self.opt_gen_xy.step()
            self.opt_gen_yx.step()
            self.gen_iterations += 1
          ###########################
          # (2) Update D networks
          ###########################
          for p in self.critic_x.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in G update
          for p in self.critic_y.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in G update              
          
          j = 0
          while j < self.CRITIC_ITERATIONS and i < len(loader):
            j += 1
            (x,y) = data_iter.next()
            i += 1
            self.critic_x.zero_grad()
            self.critic_y.zero_grad()

            critic_x_real = self.critic_x(x).view(-1)
            critic_y_real = self.critic_y(y).view(-1)
            
            fake_x = self.gen_yx(y,gen_yx_h_0)
            fake_y = self.gen_xy(x,gen_xy_h_0)
            fake_x_buff = self.gen_yx_buffer.push_and_pop(fake_x)
            fake_y_buff = self.gen_xy_buffer.push_and_pop(fake_y)
            fake_x_buff.requires_grad=True      
            fake_y_buff.requires_grad=True
            critic_x_fake = self.critic_x(fake_x_buff).view(-1)
            critic_y_fake = self.critic_y(fake_y_buff).view(-1)
            # --------------------------
            #  Train Discriminator X & Y
            # --------------------------
            # WGAN-GP
            if self.gan_mode == 'WGAN':
              # W Loss
              LossW_x = -(torch.mean(critic_x_real)  - torch.mean(critic_x_fake))
              LossW_y = -(torch.mean(critic_y_real)  - torch.mean(critic_y_fake))
              # Gradient Penalty
              GP_x = self.gradient_penalty(self.critic_x,x,fake_x_buff,
              device=self.device)*self.LAMBDA_GP_x
              GP_y = self.gradient_penalty(self.critic_y,y,fake_y_buff,
              device=self.device)*self.LAMBDA_GP_y
              # Critic Loss
              LossC_x = (LossW_x + GP_x)
              LossC_y = (LossW_y + GP_y)

            # LSGAN
            if self.gan_mode == 'LSGAN':
              #Least-Square
              LossC_x = 0.5* (
                  self.loss_gan(critic_x_real,torch.ones_like(critic_x_real)) +
                  self.loss_gan(critic_x_fake,torch.zeros_like(critic_x_fake))
                  )
              LossC_y = 0.5* (
                  self.loss_gan(critic_y_real,torch.ones_like(critic_y_real)) +
                  self.loss_gan(critic_y_fake,torch.zeros_like(critic_y_fake))
                  )

            # Summary of Discriminator
            LossW = LossW_y + LossW_x
            Loss_GP = GP_y + GP_x
            LossC = LossC_y + LossC_x

            LossC.backward()

            #grad_critic_x = self.f_norm_grad_model(self.critic_x)
            #grad_critic_y = self.f_norm_grad_model(self.critic_y)
            #writer.add_scalars('Grad_Critic/X/',grad_critic_x, epoch)
            #writer.add_scalars('Grad_Critic/Y/',grad_critic_y, epoch)

            self.opt_critic_x.step()
            self.opt_critic_y.step()

        ############################
        # End of Epoch
        ###########################  
        end_time = time.time()
        epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
        print('Epoch [{}/{}] Time: {}m {}s  Critic: {:.4f}  Gen: {:.4f}\
        W: {:.4f} GP: {:.4f} Cycle: {:.4f} GAN: {:.4f} # Idty: {:.4f}'.format(
              epoch,epoch_f,epoch_mins,epoch_secs,LossC,LossG,\
              LossW,Loss_GP,LossCycle,LossGAN,LossId)) 
        ###########################
        # Tensorboard and Save
        ###########################
        ## WEIGHTS
        #self.f_hist_weight_model('Weights_Critic/X/',self.critic_x,writer,epoch)
        #self.f_hist_weight_model('Weights_Critic/Y/',self.critic_y,writer,epoch)
        #self.f_hist_weight_model('Weights_Generator/XY/',self.gen_xy,writer,epoch)
        #self.f_hist_weight_model('Weights_Generator/YX/',self.gen_yx,writer,epoch)
        # LOSSES
        writer.add_scalars('Losses/X/', {
            'Id_x':   LossId_x.item(),
            'GAN_x':  LossGAN_x.item(),
            'Cycle_x':   LossCycle_x.item(),
            'Gen_x':   LossG_x.item(),
            'Critic_x':   LossC_x.item()}, epoch)
        writer.add_scalars('Losses/Y/', {
            'Id_y':   LossId_y.item(),
            'GAN_y':  LossGAN_y.item(),
            'Cycle_y':   LossCycle_y.item(),
            'Gen_y':   LossG_y.item(),
            'Critic_y':   LossC_y.item()}, epoch)
        writer.add_scalars('Losses/T/', {
            'Id':   LossId.item(),
            'GAN':  LossGAN.item(),
            'Cycle':   LossCycle.item(),
            'Gen':   LossG.item(),
            'Critic':   LossC.item()}, epoch)
        
        if self.gan_mode == 'WGAN':
          writer.add_scalars('Losses/X/', {
              'W_x':   LossW_x.item(),
              'GP_x':  GP_x.item()}, epoch)
          writer.add_scalars('Losses/Y/', {
              'W_y':   LossW_y.item(),
              'GP_y':  GP_y.item()}, epoch)
          writer.add_scalars('Losses/T/', {
              'W':   LossW.item(),
              'GP':  Loss_GP.item()}, epoch)
        if epoch % epoch_tb_print == 0:
        # PLOTS
          with torch.no_grad():
            self.func_plot_TB_comparison(writer,x,y,fake_x,fake_y,self.dict_x,self.dict_y,epoch)

        if epoch % epoch_save == 0:
          ## Save models
          torch.save({'epoch': epoch,
                      'gen_iterations': self.gen_iterations,
                      'gen_xy_state_dict': self.gen_xy.state_dict(),
                      'critic_y_state_dict': self.critic_y.state_dict(),
                      'opt_gen_xy_state_dict': self.opt_gen_xy.state_dict(),
                      'opt_critic_y_state_dict': self.opt_critic_y.state_dict(),
                      'gen_yx_state_dict': self.gen_yx.state_dict(),
                      'critic_x_state_dict': self.critic_x.state_dict(),
                      'opt_gen_yx_state_dict': self.opt_gen_yx.state_dict(),
                      'opt_critic_x_state_dict': self.opt_critic_x.state_dict(),                
                      'args': self.args}, '%s/Cycle_GAN_%d.pt' % (path_save, epoch))
          
      writer.close()    
      print('-----------End of Training------------')      

    def test_samples(self):
        print('------Critic_y----------')
        input = torch.randn((self.batch_size, self.CHANNELS_SIGNAL_y, self.LEN_SAMPLES)).to(self.device)
        #summary(critic, input_size=(self.CHANNELS_SIGNAL_x, self.LEN_SAMPLES))
        print(self.critic_y(input).shape)
        assert self.critic_y(input).shape == (self.batch_size, 1, 1), "Discriminator test failed"
        print('----------------------\r\n\r\n')
        
        print('------Generator_xy----------')
        input = torch.randn((self.batch_size, self.CHANNELS_SIGNAL_x, self.LEN_SAMPLES)).to(self.device)
        h_g_xy = self.gen_xy.init_hidden()
        print(h_g_xy[0].size())
        #summary(gen, input_size=(Z_DIM, 1))
        print(self.gen_xy(input,h_g_xy).shape)
        assert self.gen_xy(input,h_g_xy).shape == (self.batch_size,
        self.CHANNELS_SIGNAL_y, self.LEN_SAMPLES), "Generator test failed"
        print('----------------------\r\n\r\n')

        print('------Critic_x----------')
        input = torch.randn((self.batch_size, self.CHANNELS_SIGNAL_x, self.LEN_SAMPLES)).to(self.device)
        #summary(critic, input_size=(CHANNELS_SIGNAL, self.LEN_SAMPLES))
        print(self.critic_x(input).shape)
        assert self.critic_x(input).shape == (self.batch_size, 1, 1), "Discriminator test failed"
        print('----------------------\r\n\r\n')

        print('------Generator_yx----------')
        input = torch.randn((self.batch_size, self.CHANNELS_SIGNAL_y, self.LEN_SAMPLES)).to(self.device)
        h_g_yx = self.gen_yx.init_hidden()
        #summary(gen, input_size=(Z_DIM, 1))
        print(self.gen_yx(input,h_g_yx).shape)
        assert self.gen_yx(input,h_g_yx).shape == (self.batch_size,
        self.CHANNELS_SIGNAL_x, self.LEN_SAMPLES), "Generator test failed"
        print('----------------------\r\n\r\n')




##################################
## Main
##################################



##################################
# Dataloader
##################################
class CustomDataset(Dataset):
  def __init__(self,X,Y,device):
    self.X = X
    self.Y = Y
    self.device = device
  def __getitem__(self, index):
    x = self.X[index]
    y = self.Y[index]

    x = torch.from_numpy(x).float().to(self.device)
    y = torch.from_numpy(y).float().to(self.device)
    return (x,y)

  def __len__(self):
    return len(self.X)


##################################
# Trainer
##################################

def f_trainer(x_train,y_train,root_path,list_args,epoch_save,epoch_tb_print,epoch_f):
  """
  Args:
  -----
    x_train,y_train: {np.array} Domain X and Y to train.
    root_path: {str} Path where will be saved the models, tensorboard, and plots
    list_args: {dict} Parameters of training
    epoch_save : {int} Save every defined numbers of epochs
    epoch_tb_print: {int} Print every defined numbers of epochs
    epoch_f: {int} Number of epochs of training
  
  Return:
  -------
    None, this function do the training.
  """
  if list_args['GAN_MODE'] == 'WGAN':
    version = 'Cycle_WGAN_GP'
  elif list_args['GAN_MODE'] == 'LSGAN':
    version = 'Cycle_LSGAN'
  elif list_args['GAN_MODE'] == 'Vanilla':
    version = 'Cycle_GAN'
  else:
    raise NotImplementedError('Adversarial loss not defined')
  # Define TB, save and plot paths.
  path_dict = {}
  path_tb = root_path+version+"/logs/"
  path_dict['path_tb'] = path_tb
  plot_path = root_path+version+"/figures/"
  path_dict['plot_path'] = plot_path
  path_save = root_path+version+"/save/"
  path_dict['path_save'] = path_save

  os.makedirs(path_tb,exist_ok=True)
  os.makedirs(plot_path,exist_ok=True)
  os.makedirs(path_save,exist_ok=True)

  gc.collect()

  # Hyperparameters definitions
  lr = list_args['lr']
  batch_size = list_args['batch_size']
  LEN_SAMPLES = list_args['LEN_SAMPLES']
  RNN_GEN = list_args['RNN_GEN']
  F_GEN = list_args['F_GEN']    # RNN
  F_CRITIC = list_args['F_CRITIC'] # DCNN
  CRITIC_ITERATIONS = list_args['CRITIC_ITERATIONS'] if list_args['GAN_MODE'] is 'WGAN' else 1
  GEN_ITERATIONS = list_args['GEN_ITERATIONS'] #?
  LAMBDA_GP_x = 10 if list_args['GAN_MODE'] is 'WGAN' else 0
  LAMBDA_GP_y = 10 if list_args['GAN_MODE'] is 'WGAN' else 0
  LAMBDA_CYCLE_x = list_args['LAMBDA_CYCLE_x']
  LAMBDA_CYCLE_y = list_args['LAMBDA_CYCLE_y']
  LAMBDA_IDENTITY_x = list_args['LAMBDA_IDENTITY']
  LAMBDA_IDENTITY_y = list_args['LAMBDA_IDENTITY']
  GAN_MODE = list_args['GAN_MODE']

  CHANNELS_SIGNAL_x = list_args['CHANNELS_SIGNAL_x']
  CHANNELS_SIGNAL_y = list_args['CHANNELS_SIGNAL_y']
  device = list_args['device']

    ###############################
    # GENERATOR
    ###############################
  layers_gen_xy = {
      'in_features': CHANNELS_SIGNAL_x,
      'out_features': CHANNELS_SIGNAL_y,
      'hidden_dim': F_GEN,
      'num_layers': 2 ,
      'tanh_output':True }

  layers_gen_yx = {
      'in_features': CHANNELS_SIGNAL_y,
      'out_features': CHANNELS_SIGNAL_x ,
      'hidden_dim': F_GEN,
      'num_layers': 2 ,
      'tanh_output':True ,
  }        

    ###############################
    # CRITIC
    ###############################
  layers_critic_x = CycleGAN_helpers.f_critic_dict(CHANNELS_SIGNAL_x,F_CRITIC,LEN_SAMPLES)
  layers_critic_y = CycleGAN_helpers.f_critic_dict(CHANNELS_SIGNAL_y,F_CRITIC,LEN_SAMPLES)

  args = {
    'gan_mode':GAN_MODE,
    'lr':lr,
    'batch_size':batch_size,
    'F_GEN':F_GEN,
    'RNN_GEN': RNN_GEN,
    'F_CRITIC':F_CRITIC,
    'CRITIC_ITERATIONS':CRITIC_ITERATIONS,
    'LAMBDA_GP_x': LAMBDA_GP_x,
    'LAMBDA_GP_y': LAMBDA_GP_y,
    'LAMBDA_CYCLE_x': LAMBDA_CYCLE_x,
    'LAMBDA_CYCLE_y': LAMBDA_CYCLE_y,
    'LAMBDA_IDENTITY_x': LAMBDA_IDENTITY_x,
    'LAMBDA_IDENTITY_y': LAMBDA_IDENTITY_y,
    'layers_critic_x':layers_critic_x,
    'layers_critic_y':layers_critic_y,
    'layers_gen_xy': layers_gen_xy,
    'layers_gen_yx': layers_gen_yx,
    'device': device,
    'GEN_ITERATIONS': GEN_ITERATIONS,
    'LEN_SAMPLES':LEN_SAMPLES,
    'CHANNELS_SIGNAL_x':CHANNELS_SIGNAL_x,
    'CHANNELS_SIGNAL_y':CHANNELS_SIGNAL_y,
    'dict_x':list_args['dict_x'],
    'dict_y':list_args['dict_y'],   
    }
  # NAME DEFINITION
  current_str = ('mode={arg1},'
                 'RNN={arg2},'
                 'F_GEN={arg3},'
                 'F_CRITIC={arg4},'
                 'C_ITERATIONS={arg5},'
                 'lr={arg6},'
                 'CyclePenalty={arg7}')
  current_args = {'arg1':args['gan_mode'],
                  'arg2':args['RNN_GEN'],
                  'arg3':args['F_GEN'],
                  'arg4':args['F_CRITIC'],
                  'arg5':args['CRITIC_ITERATIONS'],
                  'arg6':args['lr'],
                  'arg7':args['LAMBDA_CYCLE_y']}
  current_name = current_str.format(**current_args)
  args['current_name'] = current_name

  # Model creation
  CycleGAN = Cycle_GAN(args).to(args['device'])
  if os.path.isdir( path_dict['path_save'] + current_name):
    print(f'Existing model: {current_name}, next ...')
    return
    # si el folder ya, no seguir entrenando
    # TODO: take the last file name/epoch and load
    #print('Loading model ...')
    #cont_training = True
    #writer_tb,current_paths = fn_chkp_tb(cont_training,current_name,CycleGAN,path_dict) 
  else:
    cont_training = False
    writer_tb,current_paths = CycleGAN_helpers.fn_chkp_tb(cont_training,current_name,CycleGAN,path_dict)
  path_save_current = current_paths['path_save_current']
  path_tb_current = current_paths['path_tb_current']
  path_plot_current = current_paths['path_plot_current']
  
  dataset = CustomDataset(x_train,y_train,device)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
  # Train
  CycleGAN.fit(epoch_f,loader,epoch_save,epoch_tb_print,writer_tb,path_save_current)
  writer_tb.close()

  return  


##################################
# Predict
##################################

def f_predict(tag_pred,Cycle_GAN, data_x, data_y):
  """
  Args:
  -----
    tag_pred: {str} 'xy' or 'yx'. Will produce signal of domain the second domain,
              given the first domain.

    Cycle_GAN: {nn.Module} Model
    data_x: {np.array} Signal of Domain X
    data_y: {np.aray} Signal of Domain Y
  
  Return:
  ------
  predictions: {Tensor} with shape torch.Size([SAMPLES, CHANNELS, LENGTH])
  """
  args = Cycle_GAN.args
  device = args['device']
  batch_size  = Cycle_GAN.batch_size
  dataset = CustomDataset(data_x,data_y,device)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  if tag_pred == 'xy':
    generator = Cycle_GAN.gen_xy
    domain_f = data_y
    input_idx = 0
  if tag_pred == 'yx':
    generator = Cycle_GAN.gen_yx
    domain_f = data_x
    input_idx = 1

  generator.eval()

  n_elements = len(dataloader.dataset)
  num_batches = len(dataloader)
  batch_size = dataloader.batch_size

  predictions = torch.zeros_like(torch.from_numpy(domain_f))

  with torch.no_grad():
    for i, batch in enumerate(dataloader):

      input_i = batch[input_idx]
      
      start = i*batch_size
      end = start + batch_size
      if i == num_batches - 1:
        end = n_elements
        batch_size = end - start

      gen_h_0 = generator.init_hidden(batch_size)
      output = generator(input_i,gen_h_0)

      predictions[start:end] = output

  return predictions


##################################
# Loader from Path
##################################

def f_load_from_path(path_save):
  """
  Args:
  ----
  path_save: {str} load_path

  Return:
  ------
  model: CycleGAN_network.Cycle_GAN instance

  """

  assert os.path.isdir(path_save) , "The path corresponding to saved models doesn't exist"
  paths =[i for i in sorted(Path(path_save).iterdir(), key=os.path.getmtime,reverse=True) if i.suffix =='.pt']

  print('Loading: ',str(paths[0]))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #Load checkpoint
  checkpoint = torch.load(paths[0],map_location=device)
  # load args saved in the checkpoint to create the model
  args = checkpoint['args']
  args['device'] = device 
  # model creation
  model = Cycle_GAN(args).to(device)
  model.epoch_i = checkpoint['epoch']
  model.gen_iterations = checkpoint['gen_iterations']
  # X->Y
  model.gen_xy.load_state_dict(checkpoint['gen_xy_state_dict'])
  model.critic_y.load_state_dict(checkpoint['critic_y_state_dict'])
  model.opt_gen_xy.load_state_dict(checkpoint['opt_gen_xy_state_dict'])
  model.opt_critic_y.load_state_dict(checkpoint['opt_critic_y_state_dict'])
  # Y->X
  model.gen_yx.load_state_dict(checkpoint['gen_yx_state_dict'])
  model.critic_x.load_state_dict(checkpoint['critic_x_state_dict'])
  model.opt_gen_yx.load_state_dict(checkpoint['opt_gen_yx_state_dict'])
  model.opt_critic_x.load_state_dict(checkpoint['opt_critic_x_state_dict'])

  return model