# -*- coding: utf-8 -*-
# loads levels into model  1-5 version contains system results from nodes
from sys import argv
import sys
import os.path
import os
from collections import defaultdict
from datetime import date
import datetime

import xlwings as xw
import pandas as pd

from sixgill.pipesim import Model
from sixgill.definitions import *

GAS_LIQ_PATTERN ={-1: 'Undefined',     0: 'Smooth',     1: 'Stratified', 
                2: 'Annular',          3: 'Slug',       4: 'Bubble', 
                5: 'Segregated',       6: 'Transition',
                7: 'Intermittent', 8: 'Distributed', 9: 'Strat. Smooth', 10: 'Strat. Wavy',
                11: 'Strat. Dispersed', 12: 'Annular Disp.', 13: 'Intermit./Slug', 14: 'Churn',
                15: 'Dispersed Bubble', 16: 'Single Phase', 17: 'Mist', 18: 'Liquid',
                19: 'Gas', 20: 'Dense Phase', 21: 'Annular Mist', 22: 'Two Phase',
                23: 'Wave', 24: 'Dispersed', 25: 'Plug', 26: 'Tr. Bubble/Slug',
                27: 'Tr. Froth/Slug', 28: 'Heading', 29: 'Oil ', 30: 'Oil/Water',
                31: 'Water/Oil', 32: 'Water', 33: 'Froth', 34: 'Strat. 3-phase',
                35: 'Bubbly', -999.0: 'None',
                }
OIL_WAT_PATTERN = {-1: 'Undefined',                 0: 'Stratified',
                1: 'Strat. liq. film in slug flow',   2: 'Oil/Water',
                3: 'Water/Oil',                       4: 'Water',
                5: 'Oil',                             6: 'Emulsion',
                -999.0: 'None',
                }


def create_result_templates():
    '''
    Creates and returns result templates, empty dataframes to fill while iterations are performing.
    '''
    results_long = pd.DataFrame(data=None, \
        columns=['Year', 'Branch', 'Pipe', 'HorizontalDistance_m', \
                'PipeInsideDiameter_mm', 'Pressure_bara', 'Pressure_atmg', 'Temperature_C', \
                'MassFlowrate_kg_s', 'VolumeFlowrateGasStockTank_mmsm3_d', \
                'VolumeFlowrateLiquidStockTank_sm3_d', 'VolumeFlowrateWaterInSitu_sm3_d', \
                'Watercut_percent', 'VelocityLiquid_m_s', 
                'VelocityGas_m_s', 'MeanVelocityFluid_m_s', 
                'FlowPatternGasLiquid', 'FlowPatternOilWater'])
    
    result_short = pd.DataFrame(data=None, 
        columns=['Year', 'Pipe', 'InnerD_mm', \
                'Thickness_mm', 'Length_km', \
                'P1_atmg', 'P2_atmg', 'dP_atm_km', \
                'T1_c', 'T2_c',])
    return result_short, results_long
    

def get_nodes_data(model):
    nds = model.find(Source=ALL)
    nds.extend(model.find(Sink=ALL))
    result_system = pd.DataFrame(data=None, index = nds, columns=None)
    return result_system
    

def get_flowlines_data(model):
    '''
    Returns refernece dictionary of flowlines' diameters, thicknesses, lengths
    '''
    flowlines = model.find(Flowline=ALL)
    f_data = {fl: {'InnerD_mm': model.get_value(fl, parameter=Parameters.Flowline.INNERDIAMETER), \
                    'Thickness_mm': model.get_value(fl, parameter=Parameters.Flowline.WALLTHICKNESS), \
                    'Length_m': model.get_geometry(fl)['MeasuredDistance'].iloc[-1]}
                    for fl in flowlines}
    return f_data
    
    
def get_short_data_row(results, flowlines_data, flowline, year, branch, point):
    '''
    Returns shot dataframe row
    '''
    if flowline in flowlines_data:
        short = pd.DataFrame(data=None, columns=['Year', 'Pipe', 'InnerD_mm', \
                                                'Thickness_mm', 'Length_km', \
                                                'P1_atmg', 'P2_atmg', 'dP_atm_km', \
                                                'T1_c', 'T2_c',])
        short.loc[0, 'Year'] = year
        short.loc[0, 'Pipe'] = flowline
        short.loc[0, 'InnerD_mm'] =     flowlines_data[flowline]['InnerD_mm']
        short.loc[0, 'Thickness_mm'] =  flowlines_data[flowline]['Thickness_mm']
        lng = flowlines_data[flowline]['Length_m']
        p1 =  results.profile[branch]['Pressure'][point]
        short.loc[0, 'Length_km'] = lng / 1000
        short.loc[0, 'P1_atmg'] = (p1 - 1.01325) / 1.01325
        short.loc[0, 'T1_c'] = results.profile[branch]['Temperature'][point]
        return short, lng, p1


def unload_results(results, year, result_short, result_long, flowlines_data: dict):
    '''
    Return two datasets: short and long. 
    Short includes only input and output  data (pressures and temepratures).
    Long incudes other different data along pipes and years. 
    '''
    for branch in results.profile:
        # result_short, result_long = create_result_templates()
        # branch = 'AGZU 2184 28.8'
        pipe = results.profile[branch]['BranchEquipment'][0]
        if pipe is None:
            pipe = ''
        else:
            short, lng, p1 = get_short_data_row(results, flowlines_data, pipe, year, branch, 0)
            result_short = result_short.append(short, ignore_index=True)
            # print('0', result_short); input()
            
        for j in range(1, len(results.profile[branch]['BranchEquipment'])):
            if (results.profile[branch]['BranchEquipment'][j] is not None) \
            or (j == len(results.profile[branch]['BranchEquipment']) - 1):
                p2 = results.profile[branch]['Pressure'][j]
                spam = result_short.shape[0] - 1
                result_short.at[spam, 'P2_atmg'] =     (p2 - 1.01325) / 1.01325
                try:
                    result_short.at[spam, 'dP_atm_km']=    (p1 - p2) / lng 
                except:
                    pass                
                result_short.at[spam, 'T2_c'] =     results.profile[branch]['Temperature'][j]
                # print(branch, j)
                
                if j != len(results.profile[branch]['BranchEquipment']) - 1: 
                    pipe = results.profile[branch]['BranchEquipment'][j]
                    short, lng, p1 = get_short_data_row(results, flowlines_data, pipe, year, branch, j)
                    result_short = result_short.append(short, ignore_index=True)
                # print(j, ": ", result_short); input()

            egg = pd.DataFrame(data=None, columns=result_long.columns)
            egg.at[0, 'Year'] =               year
            egg.at[0, 'Branch'] =             branch
            egg.at[0, 'Pipe'] =               pipe
            egg.at[0, 'HorizontalDistance_m'] =    results.profile[branch]['HorizontalDistance'][j]
            egg.at[0, 'PipeInsideDiameter_mm'] =   results.profile[branch]['PipeInsideDiameter'][j]
            egg.at[0, 'Pressure_bara'] =           results.profile[branch]['Pressure'][j]            
            # convert bara to atmg
            egg.at[0, 'Pressure_atmg'] =           results.profile[branch]['Pressure'][j]    
            egg.at[0, 'Pressure_atmg'] -= 1.01325
            egg.at[0, 'Pressure_atmg'] /= 1.01325                    
            egg.at[0, 'Temperature_C'] =           results.profile[branch]['Temperature'][j]
            egg.at[0, 'MassFlowrate_kg_s'] =       results.profile[branch]['MassFlowrate'][j]            
            egg.at[0, 'VolumeFlowrateGasStockTank_mmsm3_d'] =   results.profile[branch]['VolumeFlowrateGasStockTank'][j]
            egg.at[0, 'VolumeFlowrateLiquidStockTank_sm3_d'] =  results.profile[branch]['VolumeFlowrateLiquidStockTank'][j]            
            egg.at[0, 'VolumeFlowrateWaterInSitu_sm3_d'] =      results.profile[branch]['VolumeFlowrateWaterInSitu'][j]            
            egg.at[0, 'Watercut_percent'] =                     results.profile[branch]['Watercut'][j]
            egg.at[0, 'VelocityLiquid_m_s'] =                   results.profile[branch]['VelocityLiquid'][j]
            egg.at[0, 'VelocityGas_m_s'] =                      results.profile[branch]['VelocityGas'][j]
            egg.at[0, 'MeanVelocityFluid_m_s'] =                results.profile[branch]['MeanVelocityFluid'][j]
            try:
                egg.at[0, 'FlowPatternGasLiquid'] =       GAS_LIQ_PATTERN[results.profile[branch]['FlowPatternGasLiquid'][j]]
                egg.at[0, 'FlowPatternOilWater'] =        OIL_WAT_PATTERN[results.profile[branch]['FlowPatternOilWater'][j]]
            except Exception as ex:
                # maybe return exception out of this function
                print(ex)
                pass
            result_long = result_long.append(egg, ignore_index=True)

    return result_short, result_long
  
      
def iterate_by_years(model, levels: pd.DataFrame, save_to_separate_files=False):
    log = pd.DataFrame(data=None, columns=['Year', 'State', 'Messages'])
    for study in model.find(Study=ALL):
        if not study == 'Study 1':  #clear excessive studies
            model.delete(component=ModelComponents.STUDY, context=study)
    # create result templates
    result_short , result_long= create_result_templates()
    result_system = get_nodes_data(model)
    # get model flowlines 
    flowlines_data = get_flowlines_data(model)
    # flowlines = model.find(Flowline=ALL)
    src = model.find(Source=ALL)
    snk = model.find(Sink=ALL) 
    is_ppd = len(snk) > len(src)
    pads_model = snk if is_ppd else src
    pads_level = set(str(levels.index[i][0]) for i in range(levels.shape[0]))
    levels_lacks = [pad for pad in pads_level if pad not in pads_model]
    pads = [pad for pad in pads_level if pad in pads_model]
    # print(pads)
    # how many pads were not found?!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if levels_lacks == pads_model:
        print('В файле с уровнями не упоминаются кусты модели')
        sys.exit()
        
    boundaries = dict()  # defaultdict.fromkeys(pads)
    msg = dict()
    for year in levels.columns[1:]:
        if is_ppd:
            for pad in pads:
                boundaries[pad] = {"FlowRateType": "LiquidFlowRate",
                                   "LiquidFlowRate": levels.at[(pad, 'Закачка воды'), year], 
                                   "GOR": 0,  
                                   "WaterCut": 0,
                                   "Temperature": levels.at[(pad,'Температура'), year],
                                   }
        else:
            # year = '2011-01'
            for pad in pads:
                liq = levels.at[(pad, 'Добыча жидкости'), year]
                gas = levels.at[(pad, 'Добыча газа'), year]                    
                temp = levels.at[(pad, 'Температура'), year]

                if liq>0:
                    oil = levels.at[(pad, 'Добыча нефти'), year]
                    wat = liq - oil
                    
                    wcut = wat/liq * 100
                    gor = gas/oil if oil != 0 else 0 
                                       
                    boundaries[pad] = {"FlowRateType": "LiquidFlowRate",
                                       "LiquidFlowRate": liq, 
                                       "GOR": gor,  
                                       "WaterCut": wcut,
                                       "Temperature": temp,
                                       }
                elif gas > 0: 
                    # ПРОТЕСТИРОВАТЬ ПУСТЫЕ УРОВНИ
                    boundaries[pad] = {"FlowRateType": "GasFlowRate",
                                       "LiquidFlowRate": gas, 
                                       "GOR": 0,  
                                       "WaterCut": 0,
                                       "Temperature": temp,
                                       }
        
        # add mesasge, in how many pads 4cast data were downloaded
        # model.add(ModelComponents.STUDY, year)
        # model.tasks.networksimulation.set_conditions(boundaries=boundaries, study=year)
        # model.save()
        print(*boundaries, sep='\n')
        
        results = model.tasks.networksimulation.run(
	       system_variables = OutputVariables.System.FLOW_ASSURANCE, 
	       profile_variables = OutputVariables.Profile.FLOW_ASSURANCE, 
	       # study=year)
	       boundaries=boundaries)
        # boundaries.clear()
        print(*results.messages, sep='\n')        # of type list
        # print(*results.summary, sep='\n')        # of type dict
        print(year, 'simulation state:', results.state) # of type str, RUNNING, COMPLETED, FAILED
        print('-' * 100)
        spam = {'Year':[year for _ in range(len(results.messages))],\
                'State': [results.state for _ in range(len(results.messages))],\
                'Messages': results.messages
                }
        egg = pd.DataFrame(spam, columns=['Year', 'State', 'Messages'])
        log = log.append(egg, ignore_index=True)
        # print(spam)
        # print(egg)
        print(log)
        # msg[year] = list(results.summary)  # results.messages
        # msg[year].append({'state': results.state})        
                                        
        if save_to_separate_files:
            # path = os.path.abspath(os.path.dirname(sys.argv[0]))
            path = os.path.abspath(os.path.dirname(sys.argv[0]))
            mf = os.path.basename(model.filename)
            mf = os.path.splitext(mf)[0]
            model.save(os.path.join([path, mf, '_', year, '.pips']))
        else:
            model.save()    
        # result_short , result_long= create_result_templates()
        result_short, result_long = unload_results(results, year, result_short, result_long, flowlines_data)
        try:
            result_system[year] = pd.DataFrame(results.system).loc[result_system.index, 'SystemInletPressure']
            result_system[year] -= 1.01325
            result_system[year] /= 1.01325  # 0.980665
        except Exception as e:
            print('ошибка выгрузки данных по кустам', e)
            continue

    return result_short, result_long, result_system, msg, log
                

# lvls = pd.read_excel(io=r'D:\Psim\20211202b_y\test.xlsm',sheetname='levels', header=0, converters={'куст':str})
lvls = pd.read_excel(io=argv[2], sheet_name='levels', header=0, converters={'куст':str})
lvls = lvls.rename(columns=lambda x: str(x).replace('-01 00:00:00', ''))  # rename colunms' names
lvls = lvls.replace('nan', 0)
levels = lvls.set_index(['куст', 'Наименование'])

# print(levels)  # print('len argv', len(sys.argv))
model_file = argv[1]  # r"D:\Psim\20211202b_y\2021.pips"
# model_file = r"D:\Psim\20211202b_y\2021.pips"
model = Model.open(model_file, units=Units.METRIC)

result_short, result_long, result_system, msg, log = iterate_by_years(model, levels)
model.save()
model.close()

# print(result_short)
# print(result_long)

# xb = xw.Book(r'D:\Psim\20211202b_y\test.xlsm')  
print(argv[2])
xb = xw.Book(argv[2])
xb.sheets.add()
sht = xb.sheets.active

sht.cells(1, 4).value = result_short
sht.cells(1, 3).value = result_short
sht.cells(1, 5).value = '-'
begin = 1; end = begin+len(result_short)
sht.range(sht.cells(begin +1, 3), sht.cells(end, 3)).value = 'S'  # date.today()

begin = end + 2; end = begin + len(result_long)
sht.cells(begin, 3).value = result_long
sht.range(sht.cells(begin, 3),\
          sht.cells(end, 3)).value = 'L'  # date.today()
          
begin = end + 2; end = begin + len(result_system)
sht.cells(begin, 6).value = result_system
sht.range(sht.cells(begin, 3),\
          sht.cells(end, 3)).value = 'SystemInletPressure_atmg'  # date.today()
sht.cells(begin, 6).value = 'Pad_Name'
sht.cells(begin, 3).value = 'Value_Unit'          

begin = end + 2; end = begin + len(log)
sht.cells(begin, 3).value = log
sht.range(sht.cells(begin, 3), sht.cells(end, 3)).value = 'log'

sht.cells(1, 1).value = 'Date'
sht.cells(1, 2).value = 'Model'
sht.cells(1, 3).value = '-'
sht.range(sht.cells(2, 1),\
          sht.cells(end, 1)).value = datetime.datetime.now()  # date.today()

sht.range(sht.cells(2, 2),\
          sht.cells(end, 2)).value = model_file

# print(msg)
# sht.cells(1 + len(result_short) + 2 + len(result_long) + 2, 2).value = msg

result_short.to_csv('result_short.csv', sep='|')
result_long.to_csv('result_long.csv', sep='|')
result_system.to_csv('result_system.csv', sep='|')
log.to_csv ('log.csv', sep='|')
result_short.to_excel('result_short.xlsx', sheet_name=os.path.splitext(os.path.basename(model_file))[0])
result_long.to_excel('result_long.xlsx', sheet_name=os.path.splitext(os.path.basename(model_file))[0])
result_system.to_excel('result_system.xlsx', sheet_name=os.path.splitext(os.path.basename(model_file))[0])
log.to_excel('log.xlsx', sheet_name=os.path.splitext(os.path.basename(model_file))[0])

# model.describe(component='Source', Name='1006', parameter=Parameters.Source.LIQUIDFLOWRATE).units
# Out[145]: 'standard cubic meters per day'
