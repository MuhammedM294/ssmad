import pandas as pd
import numpy as np
from pathlib import Path

from ssmad.data_reader import *

class Aggregator:
    
    def __init__(self, df , variable=None , time_step=None , metrics=['mean']):
        self.df = df
        self.variable = variable
        self.time_step = time_step
        self.metrics = metrics
        self.grouped_df = pd.DataFrame()
        
    def _aggregate_by_month(self):
                    
        for metric in self.metrics:
            self.grouped_df[metric] = self.df.groupby([self.df.index.year, self.df.index.month]).transform(metric)
            
        return self.grouped_df
    
    def _aggregate_by_dekad(self):
        
        self.df['dekad'] = self.df.index.map(lambda x: 5 if x.day <= 10 else 15 if x.day <= 20 else 25)
        for metric in self.metrics:
            self.df[metric] = self.df.groupby([self.df.index.year,self.df.index.month,self.df['dekad']])[self.variable].transform(metric)
        self.df = self.df.drop(columns=self.variable)
        return self.df
    
    def _aggregate_by_week(self):
        
        for metric in self.metrics:
            self.grouped_df[metric] = self.df.groupby([self.df.index.year, self.df.index.isocalendar().week]).transform(metric)
            
        return self.grouped_df
    
    def _aggregate(self):
 
        if self.time_step == 'month':
            df = self._aggregate_by_month()
            return df
        
        elif self.time_step == 'dekad':
            df = self._aggregate_by_dekad()
            return self.df
        
        elif self.time_step == 'week':
            df = self._aggregate_by_week()
            return df
        
        else:
            print("Time step not supported")
            return None
    def aggregate(self):
        return self._aggregate().drop_duplicates()

class Climatology:


    def __init__(self, df, variable="sm"):
    
        self.df_original = df
        self.variable = variable
        self.df = pd.DataFrame(df[variable]).resample("1D").mean().dropna()

    @staticmethod
    def _filter_df(df=None, year=None, month=None, start_date=None, end_date=None):
  
        if df is None:
            print("No dataframe provided")
            return None
        if start_date and end_date:
            df = df.loc[start_date:end_date]
        if year:
            df = df[df.index.year == year]
        if month:
            df = df[df.index.month == month]

        return df

    def show_df(self, original=False, **kwargs):

        return self._filter_df(self.df_original, **kwargs) if original else self._filter_df(self.df, **kwargs)
    
    def _aggregate(self, time_step, metrics=['mean']):
        
        df_grouped = Aggregator(self.df, variable=self.variable, time_step=time_step, metrics=metrics)
        
        return df_grouped.aggregate()

    def compute_normal(self, time_step, metrics=['mean']):
  
        df = self._aggregate(time_step, metrics)
        for metric in metrics:
            if time_step == 'month':
                df[f"normal_{metric}"] = df.groupby(df.index.month).transform(metric)
            elif time_step == 'dekad':
                df[f"normal_{metric}"] = df.groupby([df.index.month, df['dekad']]).transform(metric)
        return df


if __name__ == "__main__":
    
    pass
    
 
   