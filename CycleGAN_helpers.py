import numpy as np
import pandas as pd
#import datetime
import time
import random,os,math,gc
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

import scipy
import scipy.io as sio
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

#Plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F

import CycleGAN_network
################################
## Helper functions
################################

def func_find_diam(area):
  """
  Args:
  -----
  area: {nd.array} Area signal

  Return:
  -------
  diameter: {nd.array} Diameter signal
  """
  diameter = 2*np.sqrt(area/np.pi)
  return diameter

def get_linreg(sigx, sigy):  
  mod, intercept, rvalue, pvalue, stderr = linregress(sigx, sigy)
  return mod, intercept  

def func_norm_all(arrays):
  input_shape = np.shape(arrays)
  assert len(input_shape)==3 , "samples should have shape [Samples , Channels , Length]"
  S,C,L = input_shape
  n_d = np.zeros_like(arrays)
  for i_s in np.arange(0,S):
    array_i = arrays[i_s]
    for i_c in np.arange(0,C):
      scaler = MinMaxScaler(feature_range = (0,1))
      min_v,max_v = array_i[i_c].min(),array_i[i_c].max()
      scaler.fit(np.array([min_v,max_v])[:, np.newaxis])
      n_d[i_s,i_c,:]  = scaler.fit_transform(arrays[i_s,i_c,:].reshape(-1,1)).reshape(-1,)
  return n_d


# To obtain number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def f_norm_grad_model(model):
# Gradient norm for the parameters other than model's bias  
# Norma del gradiente para los parametros distintos del 'bias' del modelo
  norm_grad={}
  for name, param in model.named_parameters():
    if 'bias' not in name:
      if param.requires_grad:
        norm_grad[name] = param.grad.data.norm(2).item()
  return norm_grad

def f_hist_weight_model(tag,model,writter,i):
# function that plot the histogram of each param. on tensorboard
  for name, param in model.named_parameters():
    if 'bias' not in name:
      writter.add_histogram(tag+name,param,i)
  return

################################
### Convolution Sizes
################################

# Functions to define different conv. sizes.
def func_output_size_convolution(params):
  params['output'] = float (((params['input'] - params['k'] + 2*params['p'])/params['s']) + 1)
  params['output'] = int(math.floor(params['output']))
  return params

def func_kernel_size_convolution(params):
  params['k'] = float (params['output'] -1)*params['s'] + 2*params['p'] + params['input']  
  params['k'] = int(math.floor(params['k']))
  return params

def func_input_convtranspose(params):
  params['input'] = float (( ( (params['output'] -params['k']) + 2*params['p'] ) / params['s']) + 1) 
  params['input'] = int(math.floor(params['input']))
  return params

def func_kernel_size_convtranspose(params):
  params['k'] = float (params['output'] + 2*params['p'] - params['s']*(params['input']-1))
  params['k'] = int(math.floor(params['k']))
  return params


def f_critic_dict(CHANNELS_SIGNAL,F_CRITIC,LEN_SAMPLES,k=5,s=2,p=1):
  """
  Args:
  ----
    CHANNELS_SIGNAL: Number of channels in the signal
    F_CRITIC: Number of features at the output of the first conv. layer
    LEN_SAMPLES: Length of the signal

  Returns:
  --------
    layers_critic: dictionary with all the information corresponding to each layer.

  """

  layers_critic = {
      1:{'f_in':CHANNELS_SIGNAL,  'f_out':F_CRITIC,   'input': LEN_SAMPLES , 'k': k,'s': s,'p': p, 'output':0},
      2:{'f_in':F_CRITIC ,        'f_out':F_CRITIC*2, 'input': 0 ,           'k': k,'s': s,'p': p, 'output':0},
      3:{'f_in':F_CRITIC*2 ,      'f_out':F_CRITIC*4, 'input': 0 ,           'k': k,'s': s,'p': p, 'output':0},
      4:{'f_in':F_CRITIC*4 ,      'f_out':F_CRITIC*8, 'input': 0 ,           'k': k,'s': s,'p': p, 'output':0},
      5:{'f_in':F_CRITIC*8 ,      'f_out':F_CRITIC*16,'input': 0 ,           'k': k,'s': s,'p': p, 'output':0},
      6:{'f_in':F_CRITIC*16 ,     'f_out':1,          'input': 0 ,           'k': 0,'s': s,'p': 0, 'output':1},
      }
  # Calculate inputs and outputs
  for i_l in sorted(layers_critic.keys(), reverse=False):
    if i_l < len(list(layers_critic.keys())):
      layers_critic[i_l] = func_output_size_convolution(layers_critic[i_l])
      layers_critic[i_l+1]['input'] = layers_critic[i_l]['output']
    if i_l == len(list(layers_critic.keys())):
      layers_critic[i_l] = func_kernel_size_convolution(layers_critic[i_l])
  return layers_critic       



### Plot Functions

def plot_signal_generated(signal,size,dict_label,
                          n_col =3, n_rows =3, save=False,path_name=False):
    """
    signals: {Tensor} Previously sended to cpu.
    sizes: {list} (BATCH, CHANNELS, LENGTH)
    save: {Boolean}. If image has to be saved.
    path_name: {str} path + image's name
    """
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(12, 12))
    for id_x,ax in enumerate(axes.flatten()):
      for i_c in range(size[1]): #[1]- Channels
        ax.plot(signal[id_x,i_c,:],label=dict_label[i_c])
      ax.legend()
    if save == True:
      plt.savefig(path_name)
    return fig

def plot_comparison_signal(signal_a,signal_b,size,dict_label,channel=0,
                          n_col =3, n_rows =3, save=False,path_name=False):
    """
    signals: {Tensor} Previously sended to cpu.
    sizes: {list} (BATCH, CHANNELS, LENGTH)
    save: {Boolean}. If image has to be saved.
    path_name: {str} path + image's name
    """
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(12, 12))
    for id_x,ax in enumerate(axes.flatten()):
      ax.plot(signal_a[id_x,channel,:],label='True_'+dict_label[channel])
      ax.plot(signal_b[id_x,channel,:],label='Estimated_'+dict_label[channel])
      ax.legend()
    if save == True:
      plt.savefig(path_name)
    return fig 
def plot_conditional_signal_generated(signal,labels,size,
                          n_col =3, n_rows =3, save=False,path_name=False):
    """
    signals: {Tensor} Previously sended to cpu.
    sizes: {list} (BATCH, CHANNELS, LENGTH)
    save: {Boolean}. If image has to be saved.
    path_name: {str} path + image's name
    """
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(12, 12))
    for id_x,ax in enumerate(axes.flatten()):
      for i_c in range(size[1]): #[1]- Channels
        ax.plot(signal[id_x,i_c,:])
        ax.set_title(str (labels[id_x]))
    if save == True:
      plt.savefig(path_name)
    return fig   

def func_plot_TB_comparison(writer,x,y,fake_x,fake_y,dict_x,dict_y,epoch,
                            all_domain=False):
  """
  This function plots on TB real and fake signals of the X and Y domain for each of the channels.  
  In addition, it can plot a whole domain together.

  writer: {SummaryWritter} Objet to plot on TB.
  x,y : {Tensor} Real signals.
  fake_x,fake_y: {Tensor} Transfered signals.
  dict_x,dict_y: {dict} Labels of each channels.
  epoch: {int} Training epoch.
  all_domain: {Boolean} To draw same domains togethers.
  """
  # To do for loop with zip 
  domains = ['X','Y']
  reals = [x,y]
  fakes = [fake_x,fake_y]
  dicts = [dict_x,dict_y]

  for d, r, f, dc in zip(domains,reals,fakes,dicts):
    #Channels
    for i_c in range(r.size()[1]):
      writer.add_figure('Figures/Comparacion_'+d+'/'+str(i_c),
                        plot_comparison_signal(r.cpu(),f.cpu(),r.size(),dc,channel=i_c),
                        global_step=epoch)
    if all_domain == True:
      writer.add_figure('Fig_'+d+'/Fake',
                        plot_signal_generated(f.cpu(),f.size(),dc),
                        global_step=epoch)
      writer.add_figure('Fig_'+d+'/Real',
                        plot_signal_generated(r.cpu(),r.size(),dc),
                        global_step=epoch) 
  return


def plot_cycle_1(x1,x2,cmap,s_f):
  """
  Function to plot a basic hysteresis loop with a gradient colored line referencing to time
  
  Args:
  ----
  x1: {np.ndarray} with shape (timesteps,)
  x2: {np.ndarray} with shape (timesteps,)
  cmap: {str} colormap from matplotlib
  s_f: {int} sampling frec.

  Return:
  ------
  fig: matplotlib.figure.Figure instance
  axs: {np.ndarray} with matplotlib.axes._subplots.AxesSubplot instances
  """
  t_vec = np.arange(0,len(x1)) / int (s_f)
  points = np.array([x1, x2]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  fig, axs = plt.subplots(1, 1,figsize=(7,5), sharex=True, sharey=True,squeeze=False)
  # Create a continuous norm to map from data points to colors
  norm = plt.Normalize(t_vec[0],t_vec[-1])
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  #Set the values used for colormapping
  lc.set_array(t_vec)
  lc.set_linewidth(2)
  line = axs[0,0].add_collection(lc)
  cbar = fig.colorbar(line, ax=axs[0,0])
  cbar.set_label(' Time [s]', rotation=90,fontsize=18)
  gap_lim = 0.005
  axs[0,0].set_xlim(x1.min(), x1.max())
  axs[0,0].set_ylim(x2.min(), x2.max())
  axs[0,0].set_facecolor('lightgray')  
  return fig,axs


def f_plot_cycle_2(x1,x2,fig,axs,cmap,label_cbar, s_f):
  t_vec = np.arange(0,len(x1)) / int (s_f)
  points = np.array([x1, x2]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  # Create a continuous norm to map from data points to colors
  norm = plt.Normalize(t_vec[0],t_vec[-1])
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  #Set the values used for colormapping
  lc.set_array(t_vec)
  lc.set_linewidth(2)
  line = axs.add_collection(lc)
  cbar = fig.colorbar(line, ax=axs)
  cbar.set_label(label_cbar+' T [s]')

  return fig,axs

def f_plot_cycle_3(x1,x2,fig,axs,cbar_ax,cmap,label_cbar,s_f):
  t_vec = np.arange(0,len(x1)) / int(s_f)
  points = np.array([x1, x2]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  # Create a continuous norm to map from data points to colors
  norm = plt.Normalize(t_vec[0],t_vec[-1])
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  #Set the values used for colormapping
  lc.set_array(t_vec)
  lc.set_linewidth(2)
  line = axs.add_collection(lc)
  cbar = plt.colorbar(line, cax=cbar_ax)
  cbar.set_label('T [s]',fontsize=18)
  return fig,axs

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

##############################
# Hysteresis loop plot
##############################
def f_plot_combined_cycleplot(sgn_true,sgn_pred,label_list,f_rs,fig,
                              font_size=20,cmap1='Blues',cmap2='Reds'):
  """
  Function to perform a comparison between pred and true hyst. loops.

  Args:
  -----
  sgn_true: ndarray with shape (CHANNELS, LENGTH)
  sgn_pred: ndarray with shape (CHANNELS, LENGTH)
  label_list: {list} with the name domian of each CHANELS
  f_rs: {int} sampling frec.
  fig: matplotlib.figure.Figure instance

  Return:
  -----
  fig: matplotlib.figure.Figure instance
  axs: {list} with matplotlib.axes._subplots.AxesSubplot instances
  """                              

  gs = GridSpec(1, 3, width_ratios=[30,1,1], height_ratios=[5],wspace=0.1,hspace=0.05)
  #CyclePlot
  ax0 = plt.subplot(gs[0])
  ax0.set_facecolor('grey')
  #Colorbars axes 
  ax1 = plt.subplot(gs[1])
  ax2 = plt.subplot(gs[2])
  axs_l = [ax1,ax2]
  cmap_list = [truncate_colormap(cmap1,minval=0.3,maxval = 1,n=100),
              truncate_colormap(cmap2,minval=0.3,maxval = 1,n=100)]

  # Signal label and list
  condition_list = ['True','Pred']
  sgn_list = [sgn_true,sgn_pred]
  #Limits
  x0_min = float('inf')
  x1_max = -float('inf')
  x1_min = float('inf')
  x0_max = -float('inf')

  slopes = []
  intercepts = []
  points = []
  for i, (i_cmap, i_cdn, i_sgn,ax_i) in enumerate(zip(cmap_list,condition_list,sgn_list,axs_l)):
    x_0 = i_sgn[1]
    x_1 = i_sgn[0]                                                                        
    x0_min = np.min((x0_min,np.min(x_0)))
    x1_min = np.min((x1_min,np.min(x_1)))
    x0_max = np.max((x0_max,np.max(x_0)))
    x1_max = np.max((x1_max,np.max(x_1)))
    fig,ax0 = f_plot_cycle_3(x_0,x_1,fig, ax0, ax_i, i_cmap, i_cdn,f_rs)
    slope_i, intercept_i, rvalue, pvalue, stderr = linregress(x_0, x_1)
    reg_vec_x = np.arange(-10,200)
    reg_vec_y = intercept_i + slope_i*reg_vec_x
    ax0.plot(reg_vec_x, reg_vec_y, 'k',linestyle='--',dashes=(5, 7.5),lw=1.5)
    ax_i.set_title(i_cdn)

  ax1.set_yticklabels([])
  ax1.set_ylabel('')
  ax0.set_xlim(x0_min, x0_max)
  ax0.set_ylim(x1_min, x1_max)
  return fig,(ax0,ax1,ax2)

##############################
# BLAND-ALTMAN
##############################
def bland_altman_plot(data1,data2,marca,clr='b',ax=None):
    sd_limit = 1.96
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    if ax is None:
        ax = plt.gca()
    ax.scatter(mean, diff,color=clr,alpha=0.5,marker=marca)
    ax.axhline(md,           color='r',linewidth=2)
    ax.axhline(md + sd_limit*sd, color='r', linestyle='--',linewidth=0.9)
    ax.axhline(md - sd_limit*sd, color='r', linestyle='--',linewidth=0.9)

    half_ylim = (1.5 * sd_limit) * sd
    ax.set_ylim(md - half_ylim,
                    md + half_ylim)
    # Annotate mean line with mean difference.
    ax.annotate('Mean: {:.3f}'.format(np.round(md, 3)),
                xy=(0.99, 0.55),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=12,
                xycoords='axes fraction')
    limit_of_agreement = sd_limit * sd
   # limit_of_agreement = limit_of_agreement[0]
    lower = md - limit_of_agreement
    upper = md + limit_of_agreement
    for j, lim in enumerate([lower, upper]):
        ax.annotate('-SD{}: {:.3f}'.format(sd_limit, lower),
                    xy=(0.99, 0.05),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=12,
                    xycoords='axes fraction')
        ax.annotate('+SD{}: {:.3f}'.format(sd_limit,upper),
                    xy=(0.99, 0.85),
                    horizontalalignment='right',
                    fontsize=12,
                    xycoords='axes fraction')

    return (ax)

##############################
### Inverse scale prediction
##############################
def f_inverse_scale(domain,test_sets,predictions_norm,dict_sign,reverse_scalers,f_rs):
  """
  Args:
  -----
  domain: {str} x or y
  test_sets: {dict} A dictionary with keys 'x' or 'y' containing np.array of the singla.
  predictions_norm: {ndarray} with shape (Samples, Channels, Length)
  dict_sign: {dict} i.e, :
                        { 'x': {0: 'P_Carotid', 1: 'A_Carotid'}, 
                          'y': {0: 'P_Femoral', 1: 'A_Femoral'}
                        }

  reverse_scalers: {dict} i.e.  
                  { 'y': {'P_Femoral': sklearn.preprocessing._data.MinMaxScaler instance, 
                          'A_Femoral': sklearn.preprocessing._data.MinMaxScaler instance}, 
                    'x': {'P_Carotid': sklearn.preprocessing._data.MinMaxScaler instance, 
                          'A_Carotid': sklearn.preprocessing._data.MinMaxScaler instance}
                  }
  
  Returns:
  -------
  trues: {ndarray} with shape (SAMPLES, CHANNELS, LENGTH)
  predictions: {ndarray} with shape (SAMPLES, CHANNELS, LENGTH)
  t_cycles: {ndarray} with shape (SAMPLES, 1) containing the timestep of the minimum pulse completed.
  """
  q_test = np.shape(test_sets[domain])[0]
  trues = np.zeros_like(test_sets[domain])
  predictions = np.zeros_like(predictions_norm)
  t_cycles = np.zeros((q_test,1),dtype=int)

  for i_hemo in dict_sign[domain]:
    scaler  = reverse_scalers[domain][dict_sign[domain][i_hemo]]                                                   
    for i_s in range(q_test):
      sgn_true = scaler.inverse_transform(test_sets[domain][i_s,i_hemo].reshape(-1, 1))
      sgn_pred = scaler.inverse_transform(predictions_norm[i_s,i_hemo].reshape(-1, 1))

      t_pred = np.argmin(sgn_pred[int (f_rs/2):]) + int (f_rs/2)
      t_true = np.argmin(sgn_true[int (f_rs/2):]) + int (f_rs/2)
      t_end = int (np.min((t_pred,t_true)))

      trues[i_s,i_hemo] = sgn_true.reshape(-1)
      predictions[i_s,i_hemo] = sgn_pred.reshape(-1)
      t_cycles[i_s] = t_end  

  return trues, predictions, t_cycles

##############################
### Checkpoint
##############################

def f_load_cycle(path_save,Cycle_GAN):
  """
  Arg:
  ----
    path_save: path where was saved the model
    Cycle_GAN: model
  
  Return:
  -------
  Cycle_GAN: model with the weights loaded

  xy_elements,#X-> y
  yx_elements#Y-> x
  """
  assert os.path.isdir(path_save) , "No existe el path de los modelos guardados"
  paths =[i for i in sorted(Path(path_save).iterdir(), key=os.path.getmtime,reverse=True) if i.suffix =='.pt']


  print('Loading: ',str(paths[0]))
  checkpoint = torch.load(paths[0])
  Cycle_GAN.epoch_i = checkpoint['epoch']
  Cycle_GAN.gen_iterations = checkpoint['gen_iterations']
  # X->Y
  Cycle_GAN.gen_xy.load_state_dict(checkpoint['gen_xy_state_dict'])
  Cycle_GAN.critic_y.load_state_dict(checkpoint['critic_y_state_dict'])
  Cycle_GAN.opt_gen_xy.load_state_dict(checkpoint['opt_gen_xy_state_dict'])
  Cycle_GAN.opt_critic_y.load_state_dict(checkpoint['opt_critic_y_state_dict'])
  # Y->X
  Cycle_GAN.gen_yx.load_state_dict(checkpoint['gen_yx_state_dict'])
  Cycle_GAN.critic_x.load_state_dict(checkpoint['critic_x_state_dict'])
  Cycle_GAN.opt_gen_yx.load_state_dict(checkpoint['opt_gen_yx_state_dict'])
  Cycle_GAN.opt_critic_x.load_state_dict(checkpoint['opt_critic_x_state_dict'])

  return Cycle_GAN


def fn_chkp_tb(cont_training,load_name,Cycle_GAN,path_dict):
  """
  Args:
  -----
    cont_training: Boolean, if confitnue training or not (TODO)
    load_name: name of the model
    Cycle_GAN: model
    path_dict: a dictionary with the save, tensorboard, and plot path

  Returns:
  -------
    writer_tb: the Tensorboard writter.
    current_paths: path updates (when the models doesn't exist,
                   and folders are created.)

  """

  current_paths= {}
  writter_tb = {}
  if cont_training:
    #Updating path to save
    current_paths['path_save_current'] = path_dict['path_save'] + load_name
    current_paths['path_tb_current'] = path_dict['path_tb'] + load_name
    current_paths['path_plot_current'] = path_dict['plot_path'] + load_name    
    Cycle_GAN = f_load_cycle(current_paths['path_save_current'],Cycle_GAN)
    # Load Tensorboard 
    assert os.path.isdir(current_paths['path_tb_current']) , "No existe el path del tensorboard"
  else:
    #Updating path to save
    current_paths['path_save_current'] = path_dict['path_save'] + Cycle_GAN.current_name
    current_paths['path_tb_current'] = path_dict['path_tb'] + Cycle_GAN.current_name
    current_paths['path_plot_current'] = path_dict['plot_path'] + Cycle_GAN.current_name
    # Creat dirs.
    os.makedirs(current_paths['path_save_current'],exist_ok=True)
    os.makedirs(current_paths['path_plot_current'],exist_ok=True)
    assert os.path.isdir(current_paths['path_save_current']), "No se creo el save path del modelo"
    assert os.path.isdir(current_paths['path_save_current']), "No se creo el save path del figures"
  # Creat Tensorboard
  writer_tb = SummaryWriter(f"{current_paths['path_tb_current']}")
  
  return writer_tb,current_paths


######################################
# Evaluation algorithm
######################################

def func_metrics_results(root_path,
                         test_sets,predict_set,
                         dict_sign,reverse_scalers,f_rs):
  """
  A function to evaluate the results.

  Args:
  -----
  root_path: {str}
  test_sets: {dict} i.e. (2 items) {'x': ndarray with shape (SAMPLES, CHANNELS, LENGTH), 'y': ndarray with shape (SAMPLES, CHANNELS, LENGTH)}
  predict_set: {str} 'x' or 'y' to obtain the corresponding domain.
  dict_sign: {dict} i.e. (2 items) {'x': {0: 'P_Brachial', 1: 'P_Radial'}, 'y': {0: 'P_AbdAorta', 1: 'A_AbdAorta'}}
  reverse_scalers: {dict} i.e.:
    { 'x': {'P_Radial': sklearn.preprocessing._data.MinMaxScaler instance, 'P_Brachial': sklearn.preprocessing._data.MinMaxScaler instance}, 
      'y': {'P_AbdAorta': sklearn.preprocessing._data.MinMaxScaler instance, 'A_AbdAorta': sklearn.preprocessing._data.MinMaxScaler instance}, 
      'baseline':
        { 'x': {'P_Radial': sklearn.preprocessing._data.MinMaxScaler instance, 'P_Brachial': sklearn.preprocessing._data.MinMaxScaler instance}, 
          'y': {'P_AbdAorta': sklearn.preprocessing._data.MinMaxScaler instance, 'A_AbdAorta': sklearn.preprocessing._data.MinMaxScaler instance}
        }
    }
  f_rs: {int} sampling frec.


  Return:
  -------
  df_results: {pd.DataFrame} containing the metrics and results.
  """

  df_results = {"mode":[],
                "name": [],
                "diff_Emod": [],
                "diff_Emod_p": [],
                "diff_Emod_Def": [],
                "diff_Emod_p_Def": [],           
                "RMSE_P": [],
                "RMSE_A": [],
                }
  df_results = pd.DataFrame.from_dict(df_results)
  x_test = test_sets['x']
  y_test = test_sets['y']
  q_test = np.shape(x_test)[0]

  if predict_set == 'y':
    tag_pred = 'xy'
  if predict_set == 'x':
    tag_pred = 'yx'

  # Adversarial losses (WGAN / LSGAN)
  list_modes = ["Cycle_WGAN_GP","Cycle_LSGAN"]

  for i_mode in list_modes:
    path_i_0 = root_path+i_mode+"/save/"
    path_i_0_p = root_path+i_mode+"/figures/"
    root, dirs, files = next(os.walk(path_i_0))

    for d in dirs:
      path_i_1 = path_i_0 + d
      path_i_1_p = path_i_0_p + d
      # Loading (return last trained weights)
      tested_GAN = CycleGAN_network.f_load_from_path(path_i_1)
      # Predictions
      predictions_norm = CycleGAN_network.f_predict(tag_pred,tested_GAN, x_test,y_test)
      if torch.cuda.is_available():
        predictions_norm = predictions_norm.to(torch.device("cpu"))
      predictions_norm = predictions_norm.detach().numpy()

      #Inverse scaler
      trues, predictions,t_cycles = f_inverse_scale(predict_set,test_sets,predictions_norm,dict_sign,reverse_scalers,f_rs)
      
      dict_subj = {}
      bland_altman_dict = {'P':{'min': np.zeros((q_test,2)),
                                'max': np.zeros((q_test,2))
                                },
                           'D':{'min': np.zeros((q_test,2)),
                                'max': np.zeros((q_test,2))
                                }
                           }
      for i_s in range(q_test):
        # Metrics into a dict.
        dict_subj[i_s] = {}

        t_end_i = t_cycles[i_s, 0]
        sgn_true_i = trues[i_s, :, :t_end_i] #0 - P / 1 - A
        sgn_pred_i = predictions[i_s, :, :t_end_i]
        diam_true_i = func_find_diam(sgn_true_i[1])
        diam_pred_i = func_find_diam(sgn_pred_i[1])
        # Area --> Deformation
        def_true_i = (sgn_true_i[1,:] / (sgn_true_i[1,:].min())-1)*100
        def_pred_i = (sgn_pred_i[1,:] / (sgn_pred_i[1,:].min())-1)*100        
        # Calculate metrics
        ## Pressure-Strain Elastic modulus
        mod_t, intercept_t = get_linreg(def_true_i, sgn_true_i[0])
        mod_p, intercept_p = get_linreg(def_pred_i, sgn_pred_i[0])
        diff_pmod = float(mod_t - mod_p) #Diff Peterson Module
        diff_pmod_p = (  np.abs(diff_pmod) / mod_t ) * 100
        dict_subj[i_s]['diff_Emod_Def'] = diff_pmod
        dict_subj[i_s]['diff_Emod_p_Def'] = diff_pmod_p
        # Errors - ML
        rmse_i = np.sqrt(np.square(sgn_true_i - sgn_pred_i)).mean(1) #RMSE
        mae_i = np.abs(sgn_true_i - sgn_pred_i).mean(1) #MAE
        dict_subj[i_s]['RMSE_P'] = rmse_i[0]
        dict_subj[i_s]['RMSE_A'] = rmse_i[1]
        # B-Altman
        bland_altman_dict['P']['max'][i_s] = np.array([np.max(sgn_true_i[0]),np.max(sgn_pred_i[0])])
        bland_altman_dict['P']['min'][i_s] = np.array([np.min(sgn_true_i[0]),np.min(sgn_pred_i[0])])
        bland_altman_dict['D']['max'][i_s] = np.array([np.max(sgn_true_i[1]),np.max(sgn_pred_i[1])])
        bland_altman_dict['D']['min'][i_s] = np.array([np.min(sgn_true_i[1]),np.min(sgn_pred_i[1])])        


      bland_true = [bland_altman_dict['P']['max'][:,0],
                    bland_altman_dict['P']['min'][:,0],
                    bland_altman_dict['D']['max'][:,0],
                    bland_altman_dict['D']['min'][:,0]
                    ]
      bland_pred = [bland_altman_dict['P']['max'][:,1],
                    bland_altman_dict['P']['min'][:,1],
                    bland_altman_dict['D']['max'][:,1],
                    bland_altman_dict['D']['min'][:,1]
                    ]                    
      titles = ['Systolic','Diastolic','Diam Max','Diam Min']
      unit = ['[mmHg]','[mmHg]','[cm]','[cm]']

      fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(12,10))
      faxs = axs.ravel()
      for ax_i, title_i , unit_i,true_i,pred_i in zip(faxs,titles,unit,bland_true,bland_pred):
        ax_i = bland_altman_plot(true_i,pred_i,'.',ax=ax_i)
        ax_i.set_title(title_i,fontsize=20)
        ax_i.set_ylabel('Diference '+ unit_i,fontsize=16)
        ax_i.set_xlabel('Mean '+ unit_i,fontsize=16)
      plt.subplots_adjust(hspace=0.35)
      plot_name = str(path_i_1_p+'/B_A')
      #plt.savefig(plot_name+".pdf", dpi=300,bbox_inches = "tight")
      #plt.savefig(plot_name+".svg", dpi=300,bbox_inches = "tight")
      plt.show()

      # dict --> df_i
      df_i = pd.DataFrame.from_dict(dict_subj)
      df_i = df_i.T
      # Adding columns
      df_i['mode'] = i_mode
      df_i['name'] = d
      # concatenate into only one df
      df_results = pd.concat([df_results,df_i],join='inner')

  return df_results

