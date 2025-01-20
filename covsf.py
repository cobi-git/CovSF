import argparse
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import sklearn

from module.utils import df_generate,plot,load_pickle
from module.model import CovSF

MAX_INPUT_LENGTH = 5
OUTPUT_LENGTH = 4

MM_CUT,MS_CUT = 0.2,0.5

DP_TYPES = ['Mi-M','Mi-S','M-S','Mi-M-S']
RP_TYPES = ['M-Mi','Mi-S','S-M','S-M-Mi']

PROG_SEVERITY = {'Mi-M' : ('Mild','Moderate'), 'Mi-S' : ('Mild','Severe'), 'M-S' : ('Moderate','Severe'), 'Mi-M-S' : ('Mild','Moderate','Severe'),
                'M-Mi' : ('Moderate','Mild'), 'S-Mi' : ('Severe','Mild'), 'S-M' : ('Severe','Moderate'), 'S-M-Mi' : ('Severe','Moderate','Mild')}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=str, help = 'Input file (Path) - .csv or .xlsx')
    parser.add_argument('--output',type=str, default = "./" , help = 'Path to save file')
    parser.add_argument('--name', type=str, help = 'name of save file(PDF), default = same with input file')
    args = parser.parse_args()

    file = args.input
    filename = os.path.splitext(os.path.basename(file))[0]

    save_dir = Path(args.output)

    if args.name is None: save_name = filename
    else: save_name = args.name

    # 1. Check file format in pre-process
    df,input_df = df_generate(file)

    # 2. Load model and moduels
    if torch.cuda.is_available: DEVICE = torch.device("cuda")
    else: DEVICE = torch.device("cpu")  

    model_args = load_pickle("./data/meta/model_args.pkl")
    scaler = load_pickle("./data/meta/scaler.pkl") # Robust scaler

    model = CovSF(**model_args).to(DEVICE)
    model.load_state_dict(torch.load("./data/meta/model.pt"))
    model.eval()

    # 3. Predict 
    input_days = input_df.index.tolist()
    output_days = input_days + ['+1','+2','+3']
    predicted_table = {day : [] for day in output_days}    
    for i in range(len(input_days)):
        input_idx,output_idx = input_days[max(0,i - MAX_INPUT_LENGTH + 1):i+1],output_days[i:i+OUTPUT_LENGTH]

        x = scaler.transform(input_df.loc[input_idx,].to_numpy())
        x = torch.Tensor(x).unsqueeze(0).to(DEVICE,dtype=torch.float)

        with torch.no_grad():
            predict = model(x,DEVICE)
        predict = predict.squeeze(0).cpu().numpy()[:,-1]
        
        for j,day in enumerate(output_idx):
            predicted_table[day].append(predict[j])

    # 4. Calcuate CovSF scores, classify severity and figure out severity progression
    input_days = input_df.index.tolist()
    output_days = input_days + [f'+{day}' for day in range(1,OUTPUT_LENGTH)] 
    predicted_table = {day : [] for day in output_days}    

    for i in range(len(input_days)):
        input_idx,output_idx = input_days[max(0,i - MAX_INPUT_LENGTH + 1):i+1],output_days[i:i+OUTPUT_LENGTH]

        x = scaler.transform(input_df.loc[input_idx,].to_numpy())
        x = torch.Tensor(x).unsqueeze(0).to(DEVICE,dtype=torch.float)

        with torch.no_grad():
            predict = model(x,DEVICE)
        predict = predict.squeeze(0).cpu().numpy()[:,-1]
        
        for j,day in enumerate(output_idx):
            predicted_table[day].append(predict[j])

    oxt = df['Oxygen'].tolist() + [np.nan] * 3
    covsf = np.array([np.mean(predicts) for day,predicts in predicted_table.items()])
    predict_stds = np.array([np.std(predicts) for day,predicts in predicted_table.items()])

    # Calculate DP-RP based on covSF
    pseverities = ['Mild' if score < MM_CUT else 'Moderate' if score < MS_CUT else 'Severe' for score in covsf]
    severity_sequence = [(pseverities[0],0)]
    for i,severity in enumerate(pseverities):
        if i == 0: continue
        if severity != severity_sequence[-1][0]: # severity changed
            (prev_severe,startidx) = severity_sequence.pop()
            severity_sequence.append((prev_severe,startidx,i-1)) # end previous events.
            severity_sequence.append((severity,i)) # trace new severity

    # end last severity
    (prev_severe,startidx) = severity_sequence.pop()
    severity_sequence.append((prev_severe,startidx,len(output_days)-1))

    # No CONSTANT
    pdf = pd.DataFrame({'DP' : ['x'] * len(output_days), 'RP' : ['x'] * len(output_days), 'Severity' : pseverities}, index=output_days)
    DP,RP = [],[]
    if len(severity_sequence) != 1: 
        for i, (severity,start,end) in enumerate(severity_sequence):
            if i == 0: continue
            if i >= 2:
                prevprev_severity,prevprev_start,prevprev_last = severity_sequence[i-2]
            prev_severity, prev_start, prev_last = severity_sequence[i-1]

            # Mi-M-S , S-M-Mi
            if i >= 2 and prevprev_severity == 'Mild' and prev_severity == 'Moderate' and severity == 'Severe':
                pdf.loc[output_days[prevprev_start:end+1],'DP'] = 'Mi-M-S'
            elif i >= 2 and prevprev_severity == 'Severe' and prev_severity == 'Moderate' and severity == 'Mild':
                pdf.loc[output_days[prevprev_start:end+1],'RP'] = 'S-M-Mi'

            # Mi-S ,S-Mi
            elif prev_severity == 'Mild' and severity =='Severe':
                pdf.loc[output_days[prev_start:end+1],'DP'] = 'Mi-S'
            
            elif prev_severity == 'Severe' and severity =='Mild':
                pdf.loc[output_days[prev_start:end+1],'RP'] = 'S-Mi'

            # M-S, S- M
            elif prev_severity == 'Moderate' and severity =='Severe':
                pdf.loc[output_days[prev_start:end+1],'DP'] = 'M-S'
            
            elif prev_severity == 'Severe' and severity =='Moderate':
                pdf.loc[output_days[prev_start:end+1],'RP'] = 'S-M'

            # Mi-M, M-Mi
            elif prev_severity == 'Mild' and severity =='Moderate':
                pdf.loc[output_days[prev_start:end+1],'DP'] = 'Mi-M'
            
            elif prev_severity == 'Moderate' and severity =='Mild':
                pdf.loc[output_days[prev_start:end+1],'RP'] = 'M-Mi'

    for dp_type in DP_TYPES:
        dpdf = pdf.loc[pdf['DP'] == dp_type,'Severity']
        if len(dpdf) == 0: continue
        first = current_first = PROG_SEVERITY[dp_type][0]
        last = current_last = PROG_SEVERITY[dp_type][-1]
        start_day = current_day = dpdf.index.tolist()[0]

        current = first
        for day,severity in dpdf.items():
            if current == first and severity == last: current = last
            elif current == last and severity == first:
                DP.append((dp_type,output_days.index(start_day),output_days.index(current_day)))
                current = first
                start_day = day
            current_day = day
        
        if current == last and severity == last: DP.append((dp_type,output_days.index(start_day),output_days.index(current_day)))

    for rp_type in RP_TYPES:
        dpdf = pdf.loc[pdf['RP'] == rp_type,'Severity']
        if len(dpdf) == 0: continue
        first = current_first = PROG_SEVERITY[rp_type][0]
        last = current_last = PROG_SEVERITY[rp_type][-1]
        start_day = current_day = dpdf.index.tolist()[0]

        current = first
        for day,severity in dpdf.items():
            if current == first and severity == last: current = last
            elif current == last and severity == first:
                RP.append((rp_type,output_days.index(start_day),output_days.index(current_day)))
                current = first
                start_day = day
            current_day = day

        if current == last and severity == last: RP.append((rp_type,output_days.index(start_day),output_days.index(current_day)))

    # Plot & Save
    fig = plot(
        input_days,
        oxt,
        covsf,
        predict_stds,
        pseverities,
        DP,
        RP
    )

    fig.savefig(save_dir/f'{save_name}.pdf')
