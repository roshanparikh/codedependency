import pandas as pd
import numpy as np 
from mergedata import *

# Load data
hosp_filepath_csv = "./data/mimic-iv-clinical-database-demo-2.2/hosp/csv_files/"
hosp_filepath = "./data/mimic-iv-clinical-database-demo-2.2/hosp/" 

# hm = HospitalMerge(hosp_filepath_csv)
# merged_df=hm.call(save=True, subset_frac=0.2)

# Diagnoses table
df = pd.read_csv(hosp_filepath + "diagnoses_icd.csv.gz")

# Find most common ICD-10 codes
print("\n ICD-10 \n")
df10 = df.loc[df['icd_version']==10]
icd10_df = pd.read_csv(hosp_filepath + 'd_icd_diagnoses.csv.gz')
icd10_df = icd10_df.loc[icd10_df['icd_version']==10]
df10 = pd.merge(df10, icd10_df, on="icd_code", how="left")
print(df10['long_title'].value_counts()[:15])

# Find most common ICD-9 codes
print("\n ICD-9 \n")
df9 = df.loc[df['icd_version']==9]
icd9_df = pd.read_csv(hosp_filepath + 'd_icd_diagnoses.csv.gz')
icd9_df = icd9_df.loc[icd9_df['icd_version']==9]
df9 = pd.merge(df9, icd9_df, on="icd_code", how="left")
print(df9['long_title'].value_counts()[:15])

# Filter by hypothyroidism (because why not)
