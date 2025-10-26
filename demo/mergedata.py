import pandas as pd 
import numpy as np 

class HospitalMerge():
    def __init__(self, hosp_filepath):
        self.filepath = hosp_filepath

        # Gathering tables that will be linked by subject_id. Does not include dimension tables.
        self.tables = [
            'admissions.csv', 'drgcodes.csv', 'emar.csv', 'labevents.csv', 
            'microbiologyevents.csv', 'omr.csv', 'patients.csv', 'pharmacy.csv', 
            'poe_detail.csv', 'poe.csv', 'prescriptions.csv'
            ]
        
        # Gathering dimension tables
            #NOTE: was going to incorporate these into the merged df in call, but seems clunky and inefficient
        self.dimension_tables = [
            "d_hcpcs.csv", "d_icd_diagnoses.csv", "d_icd_procedures.csv", 
            "d_labitems.csv"
            ]
    
    def call(self, save):
        '''
        Creates a csv file that merges all of the hospital demo data into two csv files.
        @params:
            save (bool): True if you want to save a csv file of the merged data without any dimension data; False otherwise.
        @returns:
            df: pandas dataframe with hospital data merged on subject_id.
        '''
        df = pd.read_csv(self.filepath + "patients.csv")
        
        for table in self.tables:
            if table != 'patients.csv':
                next_df = pd.read_csv(self.filepath + table)
                df = pd.merge(df, next_df, on="subject_id", how="left")

        if save:
            df.to_csv("merged_hosp.csv")
        
        print("Merged DataFrame shape:", df.shape)
        print(df.head())

        return df
            



