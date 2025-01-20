## CovSF
- This repository provides the implementation of the **main model (i.e CovSF)** as proposed in the submitted paper titled "*Forecasting Severity Progression Using Clinical Records and Its Application to Profiling the Single Cell Transcriptome of COVID-19 Patients*" by Bae et al.
- We provide a **terminal-based interface** for researchers, enabling the input of EHR data into the CovSF model and the retrieval of corresponding CovSF scores along with severity progression.
- The environment setup and the required format for input files (features, units, etc) are summarized below.

## Environment
- This implementation was developed primarily using **Python 3.11 and PyTorch 2.0.1**. While we provide the corresponding `environment.yml` files used for our setup, <u>users intending to leverage their GPU must install PyTorch in accordance with the compatible Python and CUDA versions.</u>

## Format for input file
- Only **.csv or .xlsx** file formats are permitted.
- **A unique index (e.g., hospitalized days) and oxygen treatment** must be included. If oxygen treatment is not available, the dataset must still contain the corresponding header.
- If the input file does not include a header, the column order must strictly conform to the table provided below.
- If header included, its name must be matched with below table.
- If headers are included, their names must exactly match those specified in the table below.

| (Order). Feature | Unit |type|
| --- | --- |---| 
| 1. Date | day |str|
| 2. BUN | 10^3/uL |float/int|
| 3. Creatinine | mg/dL |float/int|
| 4. Hemoglobin | g/dL |float/int|
| 5. LDH | U/L |float/int|
| 6. Neutrophils | 10^3/uL |float/int|
| 7. Lymphocytes | 10^3/uL |float/int|
| 8. Platelet count | 10^3/uL |float/int|
| 9. WBC Count | 10^3/uL |float/int|
| 10. CRP | mg/dL |float/int|
| 11. BDTEMP | degrees Celsius |float/int|
| 12. BREATH | BPM |float/int|
| 13. DBP | mmHg |float/int|
| 14. PULSE | BPM |float/int|
| 15. SBP | mmHg |float/int|
| 16. SPO2 | % |float/int|
| 17. Oxygen | |ROOM AIR, NASAL, MASK, HFNC, VENTILATION|
- If you are uncertain, please refer to the sample data located in `data/sample` or complete the `input_format.csv` as a guide.

## Output
- It produces patient plots analogous to Figure 2C in the main text, illustrating how the CovSF model processes the input data and delineates severity progression based on CovSF scores.

## Usage
- Just simple running code below
  
  `python covsf.py --input input_file --output output_dir --name savename`

- `--input` : **mandatory, input csv or xlsx files.**
- `--ouput` : optional, Directory where output files is saved. Default : current directory
- `--name` : optional, Name of the file to be saved. Default : Same with input file
