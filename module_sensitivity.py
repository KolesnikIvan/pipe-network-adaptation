# -*- coding: utf-8 -*-
import os.path
from collections import defaultdict, OrderedDict
import xlwings as xw
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# from pandas import read_csv, loc
# from pandas import *
from sixgill.pipesim import Model
from sixgill.definitions import *


def get_pipe_clusters(sens):
    '''
    возвращает кластеры труб на основе их влияния на источники
    '''
    mn = sens.mean(axis=1)
    sensT = sens.T  # транспонирую, чтобы кластеризовать именно трубы, как строки
    sens2T = sensT.copy()
    for idx in sensT.index:
        for cl in sensT.columns:
            sens2T[cl][idx] = 1 if sensT[cl][idx] > mn[cl] else 0      

    pipes = sens.columns  # трубы - это столбцы таблицы чувствительности
    kmeans = KMeans(n_clusters=int(len(pipes) * 0.1))
    kmeans.fit(sens2T)    
    
    clusters = kmeans.labels_
    unique_clsts = set(clusters)
    
    pipe_clusters = {kl:[] for kl in unique_clsts}
    for un in unique_clsts:
        for cl, pipe in zip(clusters, sens2T.index):
            if cl == un:
                print(cl, un, cl == un)
                pipe_clusters[un].append(pipe)
    return pipe_clusters
                
                
def pipe_sensitivity(modelfile, res_fr_csv=False, res_to_csv=True):
    """
    Estimates the sensitivity of the source inlet pressures in the given model 
    to the small friction factor insreases of each pipe.
    
    Returns orderer dict of sources containing lists of pipes, that affect the source.
    Returns list of pipes. 
    Returns dataframe of initial results.
    Returns model.
    Arguments. 
    modelfile
    res_fr_csv, res_to_csv - booleans, that specifies, if result 
    must be read to or written from a corr. file in the current directory
    """
    model = Model.open(modelfile, units=Units.METRIC) 
    model.sim_settings.use_global_flow_correlation = False
    pipes = model.find(Flowline=ALL)
    sources = model.find(Source=ALL)
    init_pars = model.sim_settings.get_flow_correlations()
    
    results = model.tasks.networksimulation.run(
        system_variables = OutputVariables.System.FLOW_ASSURANCE, 
        profile_variables = [ProfileVariables.ELEVATION])
    # model.close()
    system_df = pd.DataFrame.from_dict(results.system, orient='index')
    r0 = system_df.loc['SystemInletPressure'][sources]; system_df = None; results = None
   
    # print('pandas in dir() = ', 'pandas' in dir())
    # input()        
    if res_fr_csv:
        res = pd.read_csv('D:/Psim/202107adopt/res.csv', index_col=0, sep=';')
        # секрет был в разеделителе sep
    else:
        res = pd.DataFrame(index=sources)
        par = 1.01
        for obj in pipes:
            # model = Model.open(modelfile, units=Units.METRIC) 
            model.sim_settings.use_global_flow_correlation = False    
            new_fc_map = defaultdict(dict)    
        
            new_fc_map[obj][Parameters.FlowCorrelation.OVERRIDEGLOBAL] = True
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Horizontal.FRICTIONFACTOR] = par
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Vertical.FRICTIONFACTOR] = par
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE] = Constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION] = Constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Vertical.HOLDUPFACTOR] = 1
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE] = Constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION] = Constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED
            new_fc_map[obj][Parameters.FlowCorrelation.Multiphase.Horizontal.HOLDUPFACTOR] = 1
        
            model.sim_settings.set_flow_correlations(new_fc_map)
            new_fc_map = None
        
            print(obj)
            results = model.tasks.networksimulation.run(
            system_variables = OutputVariables.System.FLOW_ASSURANCE, 
            profile_variables = [ProfileVariables.ELEVATION])
            system_df = pd.DataFrame.from_dict(results.system, orient = 'index')
            results = None
            model.sim_settings.set_flow_correlations(init_pars)
            
            res[obj] = abs(system_df.loc['SystemInletPressure'][sources] - r0)
            system_df = None
            # model.close()
            print(pipes.index(obj), '/', len(pipes), obj, 'model closed')
        
    if res_to_csv:
        wb = xw.Book()
        wb.sheets.active.cells(1, 1).value = res
        model_name, _ = os.path.splitext(os.path.basename(modelfile))
        pth = os.path.join(os.path.dirname(modelfile), model_name, 'res.xls')
        wb.save(pth)
        wb.close()

    
    
    mn = res.mean(axis=1)  # получение средних по трубам отклонений для каджого источника
    # res['mean'] = mn
    # p_cnt = dict.fromkeys(res.columns, 0)
    # for p in res.columns:  # подсчет источников, на которые выше среднего влияет каждая труба
    #     for sr in res.index:
    #         p_cnt[p] += 1 if res[p][sr] >= mn[sr] else 0
    
    # список труб, связанных с каждым кустом
    #src_p_l = dict.fromkeys(res.index)  #, [])  
    src_p_l = {idx: [] for idx in res.index}
    for sr in res.index:
        lst = []       
        for pp in res.columns:
            if res[pp][sr] > mn[sr]:
                lst.append(pp)  
        src_p_l[sr] = lst
    # массив дебитов
    sr_fr = {i: model.get_value(i, parameter=Parameters.Source.LIQUIDFLOWRATE) for i in res.index}
    lst = sorted(list(res.index), key=sr_fr.get, reverse=True)  # сортировка по дебиту
    source_pipes_link = OrderedDict()  # обычный словарь не помнит порядок
    for k in lst:
        source_pipes_link[k] = src_p_l[k]
    
    # получаем список кустов, связанных с трубами
    pipe_src_l = {cl:[] for cl in pipes}
    for pipe in pipes:
        for src in res.index:
            if res[pipe][src] > mn[src]:
                pipe_src_l[pipe].append(src)
    
    # сортировка кластеров труб по сумме расходов связанных с ними кустов
    pipe_clusters = get_pipe_clusters(res)
    clst_wgh = dict.fromkeys(pipe_clusters.keys(), 0)
    for cl in clst_wgh:
        for pipe in pipe_clusters[cl]:
            for src in pipe_src_l[pipe]:
                clst_wgh[cl] += model.get_value(src, parameter=Parameters.Source.LIQUIDFLOWRATE)
    cl_w_l = sorted(clst_wgh, key=clst_wgh.get, reverse=True)
    cluster_s = OrderedDict()
    for clw in cl_w_l:
        cluster_s[clw] = pipe_clusters[clw]  # clst_wgh[clw]
            
    return cluster_s, pipes, r0, init_pars, model