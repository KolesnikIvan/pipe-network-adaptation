# -*- coding: utf-8 -*-
# to run with model- and xl-filenames as parameters
# python get_data.py model_file_name excel_file_name
from sys import argv
from sixgill.pipesim import Model
from sixgill.definitions import *

import xlwings as xw
import pandas as pd
import os.path
import os
 

def get_pads(model):
    src = model.find(Source=ALL)
    snk = model.find(Sink=ALL)
    pads = src if len(src) > len(snk) else snk
    # sht.cells(1, 1).value = pads
    bnd = model.tasks.networksimulation.get_conditions()
    sht.cells(1, 1).value = 'Модель'
    sht.cells(1, 2).value = model.filename
    sht.cells(2, 1).value = 'Куст'
    sht.cells(2, 2).value = 'Расход'
    sht.cells(2, 3).value = 'Расход, м3/сут'
    sht.cells(2, 4).value = 'WCut, %'
    sht.cells(2, 5).value = 'Гф, нм3/м3'
    i = 3
    for pad in pads:
        sht.cells(i,1).value = pad
        # if model.get_value(context=pad, parameter=Parameters.ModelComponent.ISACTIVE):
        if pad in bnd:
            egg = bnd[pad]['FlowRateType']
        else:
            egg = 'не активен'
        sht.cells(i,2).value = egg
        if len(src) > len(snk):
            sht.cells(i,3).value = model.get_value(context=pad, parameter=Parameters.Source.LIQUIDFLOWRATE)  # bnd[pad][egg]
            sht.cells(i,4).value = model.get_value(context=pad, parameter=Parameters.Source.WATERCUT)  # bnd[pad]['WaterCut']
            sht.cells(i,5).value = model.get_value(context=pad, parameter=Parameters.Source.GOR)  # bnd[pad]['GOR']
        else:
            sht.cells(i,3).value = model.get_value(context=pad, parameter=Parameters.Sink.LIQUIDFLOWRATE)  # bnd[pad][egg]
            sht.cells(i,4).value = model.get_value(context=pad, parameter=Parameters.Sink.WATERCUT)  # bnd[pad]['WaterCut']
            sht.cells(i,5).value = model.get_value(context=pad, parameter=Parameters.Sink.GOR)  # bnd[pad]['GOR']       

        i += 1


def get_pipes(model):
    fls = model.find(Flowline=ALL)
    sht.cells(2, 7).value = 'Т/п'
    sht.cells(2, 8).value = 'Внутренний диаметр, мм'
    sht.cells(2, 9).value = 'Толщина стенки, мм'
    sht.cells(2, 10).value = 'Длина, м'
    i = 3
    for fl in fls:
        sht.cells(i, 7).value = fl
        sht.cells(i, 8).value = model.get_value(fl, parameter=Parameters.Flowline.INNERDIAMETER)
        sht.cells(i, 9).value = model.get_value(fl, parameter=Parameters.Flowline.WALLTHICKNESS)
        egg = model.get_geometry(fl)
        sht.cells(i, 10).value = egg['MeasuredDistance'][len(egg) - 1]
        i += 1
        

# xbk = xw.books.active
# xbk.save()
# sht = xbk.sheets('id')
# sht.clear()
print(*argv)
xbk = xw.Book(argv[2])
xbk.save()
xbk.sheets.add()
sht = xbk.sheets.active


model_file = argv[1]  # r"D:\Psim\20211202b_y\2021.pips"
print(model_file)
model = Model.open(model_file, units=Units.METRIC)
get_pads(model)
get_pipes(model)
model.close()
