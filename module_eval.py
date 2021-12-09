# -*- coding: utf-8 -*-
import os  #.walk
# import os.path
os.environ['UserRestartFile'] = 'true'
from collections import defaultdict
# from time import perf_counter
import datetime
import numpy as np
import pandas as pd

from sixgill.pipesim import Model
from sixgill.definitions import *

def evaluate(x, model, params, press_t, cnt_g):
    """    
    The objective function to be minimized.
        fun(x, *args) -> float
    where x is an 1-D array with shape (n,) 
    and args is a tuple of the fixed parameters needed 
    to completely specify the function.
    """

    print('ENTERED EVAL MODULE________///////__________//////_______///')
    # init_pars = model.sim_settings.get_flow_correlations()
    #set parameters of individual into model
    model = set_vector_to_model(x, model, params)
        
    #run model and analyse fitness
    try:
        results = model.tasks.networksimulation.run(
        system_variables = OutputVariables.System.FLOW_ASSURANCE, 
        profile_variables = [ProfileVariables.ELEVATION])
        system_df = pd.DataFrame.from_dict(results.system, orient='index')  # mess_df = pd.DataFrame(results.messages)
        press_c = system_df[press_t.index].loc['SystemInletPressure']
        result = sum(abs(press_c - press_t))
        
        results = None
        system_df = None

        with open('D:/Psim/202107adopt/py/intrmres.txt', 'a', encoding='utf-8') as fl:
            # json.dump(log, fl)
            intrmres = [str(cnt_g), str(result),  datetime.datetime.now().strftime('%d.%m.%Y %H:%M')]  # str(perf_counter())]  
            intrmres.extend(map(str,x))
            intrmres.extend(params)
            intrmres = '|'.join(intrmres)
            intrmres += '\n'
            fl.writelines(intrmres)
        
        # print(cnt_g, 'rslt=', result, 'x=', x, 'prmtrs=', params, 't1=', t1, 't2=', perf_counter())

        return result
        

            
    except Exception as e:
        print(e, ' while evaluation')
        # model.sim_settings.set_flow_correlations(init_pars)        
        return float('inf')#,#3*max_TF, 


def set_vector_to_model(x, model, params):
    new_fc_map = model.sim_settings.get_flow_correlations()
    for i, fl in enumerate(params):
        new_fc_map[fl][Parameters.FlowCorrelation.Multiphase.Horizontal.FRICTIONFACTOR] = x[i]
        new_fc_map[fl][Parameters.FlowCorrelation.Multiphase.Vertical.FRICTIONFACTOR] = x[i]
        new_fc_map[fl][Parameters.FlowCorrelation.OVERRIDEGLOBAL] = True
    model.sim_settings.set_flow_correlations(new_fc_map)
    return model

def set_all_parameters_vector(model, all_parameters):
    model.sim_settings.set_flow_correlations(all_parameters)
    return model    
    
def show_corrs_by_folder(path):
    pipes = []
    for root, dirs, files in os.walk(path):
        for fl in files:
            pipes = model.find(Flowline=ALL) if pipes == [] else pipes
            model = Model.open(os.path.join(path, fl), units = Units.METRIC)
            pars = model.sim_settings.get_flow_correlations()
            for pipe in pipes:
                print(fl, '|', pipe, '|', pars[pipe][Parameters.FlowCorrelation.Multiphase.Horizontal.FRICTIONFACTOR])
            model.close()