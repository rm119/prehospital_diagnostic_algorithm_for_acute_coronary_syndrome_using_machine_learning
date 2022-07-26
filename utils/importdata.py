
import numpy as np
import pandas as pd
#import sys, os
import logging

# --- Import customized modules
#MYDIR = os.getcwd()
#sys.path.append(MYDIR+ '/utils/')
from global_variables import CORRESPONDENCE

original_file = 'data/datasheet.xlsx'
external_file = 'data/datasheet_external_cohort.xlsx'
sheet_name = 'Sheet1'


class Data():
    def __init__(self, filename=original_file):
        # import data
        try:
            self.data = pd.read_excel(filename, engine='openpyxl')
        except FileNotFoundError:
            self.data = pd.read_excel('../' + filename, engine='openpyxl')
        logging.debug("Removing unused samples.")
        self.remove_samples()
        logging.debug("Creating target columns.")
        self.set_target()
        logging.debug("Creating new columns.")
        self.create_columns()
        logging.debug("Renaming column names.")
        self.rename_columns()
        logging.debug("Replacing Unknown/Unconfirmed to Nan.")
        self.fillna_columns()
        self.replace_unknown()

    # exclude problematic samples
    def remove_samples(self):
        excluded_reasons = ['Missing diagnostic data', 'Cardiac arrest',
                            'Multiple entries (No.520)', 'Multiple entries (No.614)',
                            'Multiple entries (No.101)', 'Multiple entries (No.126) ',
                            'Multiple entries (No.608)', 'Multiple entries (No.580)',
                            'Multiple entries (No.551)', 'Multiple entries (No.339)',
                            'Multiple entries (No.376)', 'Multiple entries (No.207)',
                            'Multiple entries (No.322)', 'Multiple entries (No.301)',
                            'Multiple entries (No.30)', 'Multiple entries (No.304)',
                            'Multiple entries (No.315)', 'Multiple entries (No.223)',
                            'Multiple entries (No.135)', 'Multiple entries (No.66)',
                            'Multiple entries (No.306)']
        mask = ~self.data['reason for exclusion'].isin(excluded_reasons)
        self.data = self.data[mask].reset_index()

    # target column
    def set_target(self):
        self.data['ACS'] = self.data['clinical_disease'].isin(
            [1, 2]).astype(int)
        self.data['AMI'] = self.data['clinical_disease'].isin([1]).astype(int)
        self.data['STEMI'] = ((self.data['clinical_disease'] == 1) & (
            self.data['myocardial_infarction_type'] == 1)).astype(int)

    # target column
    def create_columns(self):
        self.data['male'] = (self.data['sex'] == 'm').astype(int)
        self.data['st_1'] = self.data.st.isin([1]).astype(int)
        self.data['st_2'] = self.data.st.isin([2]).astype(int)
        self.data['st_1_2'] = self.data.st.isin([1, 2]).astype(int)

    # fill columns with 0
    def fillna_columns(self):
        self.data['arrhythmia'] = (self.data['arrhythmia'] == 1).astype(int)
        self.data['roomair'] = (self.data['roomair'] == 1).astype(int)

    # unconfirmed to NaN:
    def replace_unknown(self):
        for i in range(1, 19):
            self.data[f'interview{i}_unconfirmed'] = (
                self.data[f'interview{i}'] == 0).astype(int)
            if i < 18:
                self.data[f'interview{i}_unknown'] = (
                    self.data[f'interview{i}'] == 1).astype(int)
            self.data.loc[self.data[f'interview{i}'].isin(
                [0]), f'interview{i}'] = np.nan
        for i in range(1, 11):
            self.data[f'medical_history{i}_unknown'] = (
                self.data[f'medical_history{i}'] == 1).astype(int)
            self.data[f'medical_history{i}_unconfirmed'] = (
                self.data[f'medical_history{i}'] == 0).astype(int)
            self.data.loc[self.data[f'medical_history{i}'].isin(
                [0, 1]), f'medical_history{i}'] = np.nan

    # for consistency with the previous datasets

    def rename_columns(self):
        self.data = self.data.rename(columns=CORRESPONDENCE)

    # known outlier
    def correct_bpmin_lt10(self):
        logging.debug("Known outliers set to Nan.")
        logging.debug(self.data.loc[(self.data.blood_pressure_min < 10) & (self.data.blood_pressure_max > 100),
                                    ['blood_pressure_min', 'blood_pressure_max']])
        self.data.loc[(self.data.blood_pressure_min < 10) & (
            self.data.blood_pressure_max > 100), 'blood_pressure_min'] = np.nan


def load_traindata(features, target):

    data = Data().data
    X = data[features]
    y = data[target]

    return X, y


def load_externaldata(features, target):

    data = Data(filename=external_file).data
    X = data[features]
    y = data[target]

    return X, y
