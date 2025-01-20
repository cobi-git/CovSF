from typing import List
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FEATURES = [
    'BUN','Creatinine','Hemoglobin','LDH','NLR','Platelet count',
    'WBC Count','CRP','BDTEMP','BREATH','DBP','PULSE','SBP','SPO2'
    ]
RAW_INPUT_FEATURES = [
    'BUN','Creatinine','Hemoglobin','LDH','Neutrophils','Lymphocytes','Platelet count',
    'WBC Count','CRP','BDTEMP','BREATH','DBP','PULSE','SBP','SPO2','Oxygen']

OUTPUT_LEN = 4

OXT_COLOR = {
    'ROOM AIR' : 'green',
    'NASAL' : 'yellow',
    'MASK' : 'lightcoral',
    'HFNC' : 'red',
    'VENTILATION' : 'purple',
    'SYMPTOM' : 'purple'
}

SEVERITY = ['Mild','Moderate','Severe']
PROG_SEVERITY = {'Mi-M' : ('Mild','Moderate'), 'Mi-S' : ('Mild','Severe'), 'M-S' : ('Moderate','Severe'), 'Mi-M-S' : ('Mild','Moderate','Severe'),
                'M-Mi' : ('Moderate','Mild'), 'S-Mi' : ('Severe','Mild'), 'S-M' : ('Severe','Moderate'), 'S-M-Mi' : ('Severe','Moderate','Mild')}
SEVERITY_COLORS = {"Mild" : "#63D7DE", "Moderate" : "#E7B800","Severe" : "#F0978D"}

MM_CUT,MS_CUT = 0.2,0.5

def df_generate(
    file_name:str,
):
    '''File format check'''
    file_type = file_name.split(".")[-1]
    if file_type not in ['csv','xlsx']:
        raise Exception("! only csv and xlsx file allowed")

    if file_type == 'csv':
        df_header = pd.read_csv(file_name,header=0)
        df_no_header = pd.read_csv(file_name,header=None)
    elif file_type == 'xlsx':
        df_header = pd.read_xlsx(file_name,header=0)
        df_no_header = pd.read_xlsx(file_name,header=None)
    else:
        raise Exception("Can't read file")

    if len(df_header.columns) != 17:
        raise Exception(
            """! Shape of Input doesn't matched with n(days) x 17
        Index is must be included.
        If you don't any oxygen information, just add header for them with feature names.
        If header exists and all header's name matched with our feature names, order of columns doesn't matter.
        If header dosen't existed, Then order of columns must be matched with us."""
        )

    if list(df_header.columns) == list(df_no_header.iloc[0]):
        df = df_header.set_index("Date")
        if sorted(list(df.columns)) != sorted(RAW_INPUT_FEATURES):
            raise Exception(f"Header names aren't matched with {RAW_INPUT_FEATURES}")
        df = df.loc[:,RAW_INPUT_FEATURES]  # Rearange
    else:
        df = df_no_header.set_index(0); df.index.name = 'Date'
        df.columns = RAW_INPUT_FEATURES

    # Interpolation
    input_df = df.loc[:,RAW_INPUT_FEATURES[:-1]]

    if not df['Neutrophils'].notnull().any() or not df['Lymphocytes'].notnull().any():
        raise Exception(
        "At least one unit of Neutropihls and Lymphocytes must be observed for NLR ratio."   
        )
    input_df = input_df.interpolate().ffill().bfill()

    input_df['NLR'] = input_df['Neutrophils'] / input_df['Lymphocytes']
    input_df = input_df.loc[:,INPUT_FEATURES]

    return (df,input_df)

def load_pickle(path):
    with open(path,'rb') as f:
        load_data = pickle.load(f)
    return load_data

def plot(
    days: List,
    oxt: List,
    covsf: np.ndarray,
    predict_stds: np.ndarray,
    pseverities:np.ndarray,
    DP:List,
    RP:List
):
    xs = [f'A+{i}' for i in range(len(days))] + [f'D+{i}' for i in range(1,OUTPUT_LEN)]

    fig, (ax,prog_ax) = plt.subplots(
        2,1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(0.8 * len(xs),8)
    )

    ax.plot(range(len(xs)),covsf,color='black',label='CovSF')
    ax.fill_between(range(len(xs)),covsf - predict_stds, covsf + predict_stds, color='gray',alpha=0.3, label='±1 SD.predict')
    ax.margins(x=0.005)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs,fontsize=13); ax.tick_params(axis='x',rotation=45)

    ax.set_ylim(-0.05,1.05)
    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=21)
    ax.set_ylabel("\nCovSF",fontsize=25)

    ploted_label = []
    for _oxt,_x,_y in zip(oxt,range(len(xs)),covsf):
        if _oxt is None or _oxt is np.nan or _oxt not in OXT_COLOR.keys(): color = 'white'; label='NaN'
        else: color = OXT_COLOR[_oxt]; label = _oxt
        if label in ploted_label: label = '_nolegend_'
        else:ploted_label.append(label)
        ax.plot([_x],[_y],linestyle='',marker='o',color=color,markersize=10, markeredgecolor='black', markeredgewidth=1,label=label)
    ax.legend()
    ax.tick_params(labelbottom=True)
    ax.grid(axis='both',alpha=0.3)

    ### Plot prgoression
    for prog,start,end in DP:
        prog_severity = pseverities[start:end+1]
        for severity in PROG_SEVERITY[prog]:
            first = prog_severity.index(severity); last = len(prog_severity) - prog_severity[::-1].index(severity)
            line, = prog_ax.plot(range(first + start,last + start + 1), [1.5] * len(range(first + start,last + start + 1)),linewidth=10, color = SEVERITY_COLORS[severity])
            line.set_solid_capstyle("round")

        prog_ax.plot(range(start,end+2),[1.5] * len(range(start,end+2)),linewidth=3,color='Red')
        prog_ax.plot([start],[1.5],marker='<',color='Red',markersize=10); prog_ax.plot([end+0.85],[1.5],marker='>',color='Red',markersize=10)
        prog_ax.text((start + end)/2, 1.4, prog,fontsize=18,fontweight='bold')
    
    for prog,start,end in RP:
        prog_severity = pseverities[start:end+1]
        for severity in PROG_SEVERITY[prog]:
            first = prog_severity.index(severity); last = len(prog_severity) - prog_severity[::-1].index(severity)
            line, = prog_ax.plot(range(first + start,last + start + 1), [0.5] * len(range(first + start,last + start + 1)),linewidth=10, color = SEVERITY_COLORS[severity])
            line.set_solid_capstyle("round")

        prog_ax.plot(range(start,end+2),[0.5] * len(range(start,end+2)),linewidth=3,color='Green')
        prog_ax.plot([start],[0.5],marker='<',color='Green',markersize=10); prog_ax.plot([end+0.85],[0.5],marker='>',color='Green',markersize=10)
        prog_ax.text((start + end)/2, 0.4, prog,fontsize=18,fontweight='bold')

    prog_ax.set_xlabel('Progression',fontsize=25)
    prog_ax.set_ylim(0,2); prog_ax.set_yticks([0.5,1.5]); prog_ax.set_yticklabels(['RP','DP'],fontsize=21)
    prog_ax.tick_params(labelbottom=False, bottom=False)
    prog_ax.tick_params(bottom=False,top=True)
    prog_ax.grid(axis='both',alpha=0.3)

    fig.subplots_adjust(hspace=0.25)   

    return fig 
