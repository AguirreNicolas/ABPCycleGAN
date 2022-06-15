import numpy as np
import scipy.io as sio

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def func_computed_x_y(arterys,hemo_variable,parametro,
                      path,f_s,n_pulses=1):
    """
    Args:
    -----
      arterys: {list} List containing the name of diff. arterial sites based
      hemo_variable: {list} List containing the associated chararcter for the requested hemovariable
      parametro: {dict} Each key contain a list of the requested parameters of the DB. Parameters are splited according the DB, and each category is the key of the dict.
      path: {str} Path to pwdb_data.mat file
      f_s: {int} sampling frecuency
      n_pulses: {int} number of copies of ech pulses.

    Return:
    -------
      db_dict: {dict} A dictionary where 'x' key correspond to signals and 'y' correspond to features/parameters.
       Furtheremore, many others dict that will help during the processing are added.
    """    
                          
    data = loadmat(path)

    plausibility = data['data']['plausibility']['plausibility_log']
    valids = [i for i, x in enumerate(plausibility) if x == 1]

    q_arteries = len(arterys)
    q_hemo_var = len(hemo_variable)
    q_channels = q_arteries*q_hemo_var

    dict_waves = {}
    dict_hemo = {}
    dict_artery = {}
    for i_h in hemo_variable:
        dict_hemo[i_h] = []
    for i_h in arterys:
        dict_artery[i_h] = []

    idx_chn = 0
    for i_art in arterys:
        for i_h in hemo_variable:
            name = i_h + '_' + i_art
            dict_hemo[i_h].append(idx_chn)
            dict_artery[i_art].append(idx_chn) 
            dict_waves[name] = [idx_chn]
            idx_chn = idx_chn + 1
    dict_waves_reverse = {v[0]: k for k, v in dict_waves.items()}

    x = []
    durations = []
    y = []

    for i_v in valids:
        # to X
        signal = []
        wave_durations = []
        for i_waves in dict_waves.keys():
            if i_waves[0] =='Q':
              wave_A = data['data']['waves']['A'+i_waves[1:]][i_v]*n_pulses
              wave_U = data['data']['waves']['U'+i_waves[1:]][i_v]*n_pulses            
              # element-wise multiplication of two lists
              wave = [a*b for a,b in zip(wave_A,wave_U)]
              wave_durations.append(len(wave))
              onset = int(data['data']['waves']['onset_times']['U'+i_waves[1:]][i_v]*f_s)
              wave = wave[-onset:]+wave[:-onset]
              signal.append(wave)
            else:  
              wave = data['data']['waves'][i_waves][i_v]*n_pulses
              wave_durations.append(len(wave)) 
              onset = int(data['data']['waves']['onset_times'][i_waves][i_v]*f_s)
              wave = wave[-onset:]+wave[:-onset]
              signal.append(wave)
        x.append(signal)
        durations.append(wave_durations)

        feature = []
        # to Y
        if 'haemods' in list(parametro.keys()):
            f1 = data['data']['haemods']
            for i in parametro['haemods']:
                f2 = f1[i_v][i]
                feature.append(f2)
        if 'pw_inds' in list(parametro.keys()):
            f1 = data['data']['pw_inds'][i_v]
            for i in parametro['pw_inds']:
                f2 = f1[i]
                feature.append(f2)
        if 'config' in list(parametro.keys()):
            f1 = data['data']['config']
            for i in parametro['config']:
                f2 = f1[i][i_v]
                feature.append(f2)
        y.append(feature)
    q_files = len(y)
    durations = np.asarray(durations)
    y = np.asarray(y)

    db_dict = {}
    db_dict['x'] = x
    db_dict['y'] = y
    db_dict['q_arteries'] = q_arteries
    db_dict['q_hemo_var'] = q_hemo_var
    db_dict['q_files'] = q_files
    db_dict['q_channels'] = q_channels
    db_dict['durations'] = durations
    db_dict['dict_hemo'] = dict_hemo
    db_dict['dict_waves'] = dict_waves
    db_dict['dict_waves_reverse'] = dict_waves_reverse
    db_dict['dict_artery'] = dict_artery

    return db_dict




#Estas funciones definen como el modo en que terminan las señales mas cortas:

# repeat_cycle: se repite desde el principio de la señal
# repeat_value: se repite el ultimo valor por la diferencia de tiempo que dure el ciclo (delta)

def func_repeat_cycle(x,durations,q_files,q_channels):
  max_dur = np.max(durations) 
  final_x = np.zeros((q_files,q_channels,max_dur))
  for i_s in np.arange(0,q_files):
    for i_c in np.arange(0,q_channels):
      if durations[i_s,i_c] < max_dur:
        end = durations[i_s,i_c]
        delta = max_dur - end
        final_x[i_s,i_c,:end] = np.array(x[i_s][i_c])
        final_x[i_s,i_c,-delta:] = np.array(x[i_s][i_c][0:delta])
      else:
        final_x[i_s,i_c] = x[i_s][i_c]
  return(final_x,max_dur)
def func_repeat_value(x,durations,q_files,q_channels):
  max_dur = np.max(durations)
  final_x = np.zeros((q_files,q_channels,max_dur))
  for i_s in np.arange(0,q_files):
    for i_c in np.arange(0,q_channels):
      if durations[i_s,i_c] < max_dur:
        end = durations[i_s,i_c]
        delta = max_dur - end
        value_end = x[i_s][i_c][end-1]
        final_x[i_s,i_c,:end] = np.array(x[i_s][i_c])
        final_x[i_s,i_c,-delta:] = np.reshape(np.repeat(value_end, delta, axis=0),(-1,delta))
      else:
        final_x[i_s,i_c] = x[i_s][i_c]
  return(final_x,max_dur)

def func_cycle_end(key,x,durations,q_files,q_channels):
  switcher = {
        'repeat_cycle': func_repeat_cycle,
        'repeat_value': func_repeat_value
        }
  # Get the function from switcher dictionary
  func = switcher.get(key, lambda: "Invalid parameter")
  # Excecute
  x,max_dur = func(x,durations,q_files,q_channels)
  return(x,int(max_dur))

  