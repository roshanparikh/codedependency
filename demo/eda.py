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

# # Find most common ICD-10 codes
# print("\n ICD-10 \n")
# df10 = df.loc[df['icd_version']==10]
# icd10_df = pd.read_csv(hosp_filepath + 'd_icd_diagnoses.csv.gz')
# icd10_df = icd10_df.loc[icd10_df['icd_version']==10]
# df10 = pd.merge(df10, icd10_df, on="icd_code", how="left")
# print(df10['long_title'].value_counts()[:15])

# # Find most common ICD-9 codes
# print("\n ICD-9 \n")
# df9 = df.loc[df['icd_version']==9]
# icd9_df = pd.read_csv(hosp_filepath + 'd_icd_diagnoses.csv.gz')
# icd9_df = icd9_df.loc[icd9_df['icd_version']==9]
# df9 = pd.merge(df9, icd9_df, on="icd_code", how="left")
# print(df9['long_title'].value_counts()[:15])

# # Filter by hypothyroidism (ICD-10: E039) (because why not)
# df_ht = df.loc[df['icd_code']=='E039']

# # Joining with some other data
# next_df = pd.read_csv(hosp_filepath + 'admissions.csv.gz')
# next_df = next_df.drop('subject_id', axis=1)
# df_ht = pd.merge(df_ht, next_df, on="hadm_id", how="left")

# next_df = pd.read_csv(hosp_filepath + 'prescriptions.csv.gz')
# next_df = next_df.drop('subject_id', axis=1)
# df_ht = pd.merge(df_ht, next_df, on="hadm_id", how="right")

# print(df_ht.columns)
# print(df_ht.loc[df_ht['drug_type']=='MAIN']['drug'].value_counts())

# Function to do the above for any ICD code
def filter_for_drugs(icd, df):
    df_filt = df.loc[df['icd_code']==icd]

    # Joining with some other data
    next_df = pd.read_csv(hosp_filepath + 'admissions.csv.gz')
    next_df = next_df.drop('subject_id', axis=1)
    df_filt = pd.merge(df_filt, next_df, on="hadm_id", how="left")

    next_df = pd.read_csv(hosp_filepath + 'prescriptions.csv.gz')
    next_df = next_df.drop('subject_id', axis=1)
    df_filt = pd.merge(df_filt, next_df, on="hadm_id", how="left")

    print(df_filt.columns)
    print(df_filt.loc[df_filt['drug_type']=='MAIN']['drug'].value_counts())

    return df_filt

df_arthero = filter_for_drugs(icd='I2510', df=df)
df_ht = filter_for_drugs(icd='E039', df=df)

# Next step: look at ICD-10s by outcome (like fraction of ppl who died)

