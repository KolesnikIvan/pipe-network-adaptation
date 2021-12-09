# -*- coding: utf-8 -*-
import sys
import json
import datetime
import numpy as np
from time import perf_counter
import pandas as pd
# from pandas import *
from scipy.optimize import minimize

import module_sensitivity
import module_eval


# логировать целевой функции ?декоратора; # fix time # передать модель, а не адрес
# как хранить промежуточные результаты # в функцию передавать кусты, по которым оценка

# Подготовка логов
with open('D:/Psim/202107adopt/py/intermediate_results.txt', 'w', encoding='utf-8') as fl:
    fl.writelines(str(datetime.datetime.now()) + '\n')
with open('D:/Psim/202107adopt/py/log.txt', 'a', encoding='utf-8') as fl:    
    fl.writelines(str(datetime.datetime.now()) + '\n')        
with open('D:/Psim/202107adopt/py/result_vector.txt', 'w', encoding='utf-8') as fl:    
    fl.writelines(str(datetime.datetime.now()) + '\n')        
        
# Получение модели, исходных и целевых параметров
method = 'nelder-mead'
modelfile = "D:/Psim/202107adopt/models/ArlanATBO10.pips"
press_t0 = pd.read_csv('D:/Psim/202107adopt/py/ATBO10prssrsPNT.csv', 
                       names=['SystemInletPressure'], 
                       index_col=0, sep=';')  # target pressures
press_t = press_t0.squeeze()                       

pipes_clusters, pipes, _, init_pars, model = \
    module_sensitivity.pipe_sensitivity(modelfile, res_fr_csv=True, res_to_csv=False)
log = dict()
cnt = 0

for cl in pipes_clusters:
    # n_par = len(source_pipes_links[source]) #len(pipes)
    # bounds=[[0.3, 7] for _ in range(n_par)]
    # cons = ()
    # for i in range(n_par):
    #     cons = cons + ({'type':'ineq', 'fun':lambda params: params[i] - 0.3},)   
    # x0 = [init_pars[pipe][Parameters.FlowCorrelation.Multiphase.Horizontal.FRICTIONFACTOR] for pipe in pipes]  # np.ones((1, n_par))#array([random.uniform(0.8, 1.2)] * n_par)
    x0 = [init_pars[pipe]['HorizontalMultiphaseFrictionFactor'] for pipe in pipes_clusters[cl]]
    res = minimize(module_eval.evaluate, 
                x0, 
                args=(model, pipes_clusters[cl], press_t, cnt),
                method = method,
                # bounds = bounds, 
                # callback = output, 
                options = {'maxiter' : 20,  # 20
                            'fatol' : 1e-3,
                            'disp' : True, 
                            'xatol' : 1e-2})  # was 1e-8, 
                            # 'adaptive' : True, 
                            # 'return_all': True})
    
    # перенос результата вычислений в модель и в лог
    for i, pipe in enumerate(pipes_clusters[cl]):
        init_pars[pipe]['HorizontalMultiphaseFrictionFactor'] = res.x[i]
        init_pars[pipe]['OverrideGlobal'] = True
    model = module_eval.set_all_parameters_vector(model, init_pars) # здесь был только вызов, не хватало присваивания: модель не обновлялась
    
    with open('D:/Psim/202107adopt/py/result_vector.txt', 'a', encoding='utf-8') as fl:    
        x = [cnt, datetime.datetime.now().strftime('%d.%m.%Y %H:%M'), method, str(res.fun), str(res.message)]
        x.extend([str(init_pars[pipe]['HorizontalMultiphaseFrictionFactor']) for pipe in pipes])
        fl.writelines('|'.join(x) + '\n')
    # вариант с сохранением промежуточных моделей
    # model = module_eval.set_vector_to_model(res.x, model,source_pipes_links[source])
    # model.save('D:/Psim/202107adopt/models/ATBO10neldermead/ATBO10' + str(cnt) + '.pips')    
        
    tm = str (perf_counter())
    log[cnt] = {'cnt': cnt, 'method': method, 'res': res, 'cluster': pipes_clusters[cl], 'vecotr':res.x, 'time': tm}
    cnt += 1
    print('log ', sys.getsizeof(log))
    with open('D:/Psim/202107adopt/py/log.txt', 'a', encoding='utf-8') as fl:
        fl.writelines(str(log) + '\n')
    log.clear()
    # if sys.getsizeof(log) >= 1e03:    # if len(log) >= 100: 
    # with open('D:/Psim/202107adopt/py/res.json/', 'w', encoding='utf-8') as fl:
    #     json.dump(res, fl)        

for pipe in pipes:
    print(pipe, init_pars[pipe]['HorizontalMultiphaseFrictionFactor'])
model.save("D:/Psim/202107adopt/models/" + method + ".pips")

                
# with open('D:/Psim/202107adopt/py/res_l.json/', 'w', encoding='utf-8') as fl:
#     json.dump(res, fl)    
# print('end')                         

# scipy.optimize.minimize(fun, 
    # x0, 
    # args=(), 
    # method='Nelder-Mead', 
    # bounds=None, 
    # tol=None, 
    # callback=None, 
    # options={'func': None, 
        # 'maxiter': None, 
        # 'maxfev': None, 
        # 'disp': False,              # disp Set to True to print convergence messages.
        # 'return_all': False, 
        # 'initial_simplex': None, 
        # 'xatol': 0.0001, 
        # 'fatol': 0.0001, 
        # 'adaptive': False})
# Options
# -------
# disp : bool
#     Set to True to print convergence messages.
# maxiter, maxfev : int
#     Maximum allowed number of iterations and function evaluations.
#     Will default to ``N*200``, where ``N`` is the number of
#     variables, if neither `maxiter` or `maxfev` is set. If both
#     `maxiter` and `maxfev` are set, minimization will stop at the
#     first reached.
# initial_simplex : array_like of shape (N + 1, N)
#     Initial simplex. If given, overrides `x0`.
#     ``initial_simplex[j,:]`` should contain the coordinates of
#     the j-th vertex of the ``N+1`` vertices in the simplex, where
#     ``N`` is the dimension.
# xatol : float, optional
#     Absolute error in xopt between iterations that is acceptable for
#     convergence.
# fatol : number, optional
#     Absolute error in func(xopt) between iterations that is acceptable for
#     convergence.
        
#    res = None
#    res = minimize(evaluate, 
#                x0, 
#                method = 'Powell', 
#                options = {'disp' : True, 
#                            'xtol':0.0001,
#                            'ftol':0.0001,
#                            'maxfev' : 400,
#                            'direc':direc, 
#                            'return_all':True, 
#                            'bounds' : bounds})