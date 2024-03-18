import random
from unittest.mock import Mock
import pytest
import pandas as pd
from ssmad.climatology import Aggregator, MonthlyAggregator,\
                              DekadalAggregator, WeeklyAggregator,\
                              BimonthlyAggregator, DailyAggregator,\
                              Climatology
                              


class TestAggregator:
    """
    Class for testing the Aggregator class.
    """
           
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of Aggregator for testing.
        """
        return Aggregator(df = data_sample, variable="sm", \
                          timespan=['2022-01-01', '2022-12-31'])
        
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the Aggregator class.
        """
        assert aggregator is not None
        assert aggregator.original_df.equals(data_sample)
        assert aggregator.variable == "sm"
        assert aggregator.timespan == ['2022-01-01', '2022-12-31']
        
    def test_validation(self, data_sample, _class = Aggregator):
        """
        Test validation of input parameters.
        """
        
        # Test for invalid input parameter for the pandas DataFrame
        with pytest.raises(TypeError):
            _class([], "sm", "month" )
        
        # Test for invalid input parameter for the pandas DataFrame not having a datetime index
        with pytest.raises(ValueError):
            _class(pd.DataFrame(), "sm")
         
        # Test for invalid input parameter for the variable   
        with pytest.raises(ValueError):
            _class(data_sample, "invalid_column" )
            
        
        
    def test_smoothing(self, data_sample, _class = Aggregator , variable = "sm"):
        """
        Test smoothing of Aggregator class.
        """
        df_without_smoothing = _class(data_sample, "sm", fillna=True,\
                                     fillna_window_size=3, smoothing=False)\
                                     .aggregate(start_date='2022-01-15', end_date='2022-01-27')
        df_with_smoothing = _class(data_sample, "sm", fillna=True,\
                                   fillna_window_size=3, smoothing=True, smooth_window_size=11)\
                                   .aggregate(year = 2022, month = 1, day = 21)
        
        # Check if the new value of the variable is the mean of the original values within the window
        assert df_without_smoothing[variable].mean() == pytest.approx(df_with_smoothing[variable].values[0] , rel=1e-1)
        
        
    
    @pytest.mark.skip(reason="Test implemented only for child classes.")
    def test_aggregate(self, aggregator, variable = "sm"):
        """
        Test the aggregation method of Aggregator class.
        """
        
        assert aggregator.aggregate() is not None
        # Perform aggregation       
        df = aggregator.aggregate()
        # Check if aggregation result is a DataFrame
        assert isinstance(df, pd.DataFrame)
        assert f"{variable}-avg" in df.columns
     
    @pytest.mark.skip(reason="Test implemented only for child classes.")    
    def test_drop_duplicates(self, aggregator):
        """
        Test the drop_duplicates of Aggregator class.
        """
        # Perform aggregation
        resulted_df = aggregator.aggregate()
        # Check if aggregation result has no duplicates
        assert len(resulted_df) == len(resulted_df.drop_duplicates())
        
        
   
class TestMonthlyAggregator(TestAggregator):
    """
    Class for testing the MonthlyAggregator class.
    """
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of MonthlyAggregator for testing.
        """
        return MonthlyAggregator(df = data_sample, variable="sm",\
                                timespan=['2022-01-01', '2022-12-31'])
    
    
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the MonthlyAggregator class.
        """
        super().test_initialization(aggregator, data_sample)
        
    
    def test_aggregate(self, data_sample ,aggregator, variable = "sm"):
        """
        Test the aggregation method of MonthlyAggregator class.
        """
        super().test_aggregate(aggregator, variable)
        df = aggregator.aggregate(month = 1)
        assert len(df) == 1
        
        daily_obs = Aggregator(data_sample, variable).aggregate(year = 2022, month = 2)
        month_avg = MonthlyAggregator(data_sample, variable).aggregate(year = 2022, month = 2)    
        
        assert daily_obs[variable].mean() == pytest.approx(month_avg[f"{variable}-avg"].iloc[0])
        
        
    def test_drop_duplicates(self , aggregator):
        """
        Test the drop_duplicates of MonthlyAggregator class.
        """
        super().test_drop_duplicates(aggregator)
        
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of MonthlyAggregator class with filtering.
        """
        
        # Perform aggregation  and filter the result to include only those from the year 2022
        resulted_df = aggregator.aggregate(year = 2022)
        # Check if aggregation result has 12 rows (12 months in a year)
        assert len(resulted_df) == 12
        
        # Perform aggregation  and filter the result to include only those from the year 2022 and January
        resulted_df = aggregator.aggregate(year = 2022, month = 1)
        # Check if aggregation result has 1 row (January, 2022)
        assert len(resulted_df) == 1
        

class TestDekadalAggregator(TestAggregator):
    
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of DekadalAggregator for testing.
        """
        return DekadalAggregator(df = data_sample, variable="sm"\
            , timespan=['2022-01-01', '2022-12-31'])
    
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the DekadalAggregator class.
        """
        super().test_initialization(aggregator, data_sample)
        
    def test_aggregate(self,data_sample ,aggregator, variable = "sm"):
        """
        Test the aggregation method of DekadalAggregator class.
        """
        super().test_aggregate(aggregator, variable)
        df = aggregator.aggregate()
        
        
        daily_obs = Aggregator(data_sample, variable).aggregate(year = 2019 , month = 1, dekad = 3)
        dekadal_avg = DekadalAggregator(data_sample, variable).aggregate(year = 2019 , month = 1, dekad = 3)
        
        assert daily_obs[variable].mean() == pytest.approx(dekadal_avg[f"{variable}-avg"].iloc[0])
        
    def test_drop_duplicates(self , aggregator):
        """
        Test the drop_duplicates of DekadalAggregator class.
        """
        super().test_drop_duplicates(aggregator)
    
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of DekadalAggregator class with filtering.
        """
        
        df = aggregator.aggregate(year = 2022, month = 1, dekad = 1)
        # Check if aggregation result has 1 row (January, 2022, dekad 1)
        assert len(df) == 1
        
        df = aggregator.aggregate(year = 2022, dekad = 2)
        # Check if aggregation result has 12 rows (12 second dekads in a year)
        assert len(df) == 12
        
        with pytest.raises(ValueError):
            aggregator.aggregate(year = 2022, month = 1, dekad = 4)
        

        

class TestWeeklyAggregator(TestAggregator):
    
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of WeeklyAggregator for testing.
        """
        return WeeklyAggregator(df = data_sample, variable="sm",
            timespan=['2022-01-01', '2022-12-31'])
    
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the WeeklyAggregator class.
        """
        super().test_initialization(aggregator, data_sample)
    
    def test_aggregate(self,data_sample, aggregator, variable = "sm"):
        """
        Test the aggregation method of WeeklyAggregator class.
        """
        super().test_aggregate(aggregator, variable)
        
        daily_obs = Aggregator(data_sample, variable).aggregate(year = 2015 , week = 15)
        weekly_obs = WeeklyAggregator(data_sample, variable).aggregate(year = 2015 , week = 15)
        
        assert daily_obs[variable].mean() == pytest.approx(weekly_obs[f"{variable}-avg"].iloc[0])
    
    def test_drop_duplicates(self , aggregator, variable = "sm"):
        """
        Test the drop_duplicates of WeeklyAggregator class.
        """
        super().test_drop_duplicates(aggregator)
    
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of WeeklyAggregator class with filtering.
        """
        
        resulted_df = aggregator.aggregate(year = 2022, week = 1)
        # Check if aggregation result has 1 row (week number 1 in 2022)
        assert len(resulted_df) == 1
        
        resulted_df = aggregator.aggregate(year = 2022)
        # Check if aggregation result has 52 rows (52 weeks in a year)
        assert len(resulted_df) == 52
        

class TestBimonthlyAggregator(TestAggregator):
    """
    Class for testing the BimonthlyAggregator class.
    """
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of BimonthlyAggregator for testing.
        """
        return BimonthlyAggregator(df = data_sample, variable="sm",
            timespan=['2022-01-01', '2022-12-31'])
    
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the BimonthlyAggregator class.
        """
        super().test_initialization(aggregator, data_sample)
    
    def test_aggregate(self,data_sample, aggregator, variable = "sm"):
        """
        Test the aggregation method of BimonthlyAggregator class.
        """
        super().test_aggregate(aggregator, variable)
        
        daily_obs = Aggregator(data_sample, variable).aggregate(year = 2015 , month = 5, bimonth = 1)
        bimonthly_obs = BimonthlyAggregator(data_sample, variable).aggregate(year = 2015 ,month = 5,  bimonth = 1)
        
        assert daily_obs[variable].mean() == pytest.approx(bimonthly_obs[f"{variable}-avg"].iloc[0])
    
    def test_drop_duplicates(self , aggregator, variable = "sm"):
        """
        Test the drop_duplicates of BimonthlyAggregator class.
        """
        super().test_drop_duplicates(aggregator)
    
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of BimonthlyAggregator class with filtering.
        """
        
        resulted_df = aggregator.aggregate(year = 2022, bimonth = 1)
        # Check if aggregation result has 12 rows (12 first bimonth 2022)
        assert len(resulted_df) == 12
        
        resulted_df = aggregator.aggregate(year = 2022)
        # Check if aggregation result has 24 rows (24 bimonths in a year)
        assert len(resulted_df) == 24
        


class TestDailyAggregator(TestAggregator):
    """
    Class for testing the DailyAggregator class.
    """
    
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of DailyAggregator for testing.
        """
        return DailyAggregator(df = data_sample, variable="sm", 
                              timespan=['2022-01-01', '2022-12-31'])
        
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the DailyAggregator class.
        """
        super().test_initialization(aggregator, data_sample)
    
    def test_aggregate(self,data_sample, aggregator, variable = "sm"):
        """
        Test the aggregation method of DailyAggregator class.
        """
        super().test_aggregate(aggregator, variable)
        
        daily_obs = Aggregator(data_sample, variable).aggregate(year = 2015 , month = 5, day = 1)
        daily_avg = DailyAggregator(data_sample, variable).aggregate(year = 2015 , month = 5, day = 1)
        
        assert daily_obs[variable].mean() == pytest.approx(daily_avg[f"{variable}-avg"].iloc[0])    

    
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of DailyAggregator class with filtering.
        """
        
        resulted_df = aggregator.aggregate(year = 2022, month = 1, day = 1)
        # Check if aggregation result has 1 row (January 1, 2022)
        assert len(resulted_df) == 1
    


class TestClimatology:
    """
    Class for testing the Climatology class.
    """
    
    @pytest.fixture
    def climatology(self, data_sample):
        """
        Fixture to create an instance of Climatology for testing.
        """
        return Climatology(df = data_sample, variable="sm", time_step='month',\
             normal_metrics=['mean']
            )
        
    @pytest.fixture
    def clim_metrics_all(self, data_sample):
        """
        Fixture to create an instance of Climatology for testing.
        """
        return Climatology(df = data_sample, variable="sm", time_step='month',\
             normal_metrics=['mean','median','min','max']
            )
        
        
    def test_initialization(self, climatology, data_sample):
        """
        Test the initialization of the Climatology class.
        """
        assert climatology is not None
        assert climatology.original_df.equals(data_sample)
        assert climatology.variable == "sm"
        assert climatology.fillna == False
        assert climatology.fillna_window_size == None
        assert climatology.smoothing == False
        assert climatology.smooth_window_size == None
        assert climatology.timespan == None
        assert climatology.time_step == "month"
        assert climatology.normal_metrics == ['mean']
        
        
        
    def test_validation(self, data_sample, _class = Climatology):
        """
        Test validation of input parameters.
        """
        
        # Test for invalid input parameter for the pandas DataFrame
        with pytest.raises(TypeError):
            _class([], "sm", "month" , ['mean'] )
        
        # Test for invalid input parameter for the pandas DataFrame not having a datetime index
        with pytest.raises(ValueError):
            _class(pd.DataFrame(), "sm" , "month" , ['mean'])
         
        # Test for invalid input parameter for the variable   
        with pytest.raises(ValueError):
            _class(data_sample, "invalid_column" , "month" , ['mean'])
            
        # Test for invalid input parameter for the time_step   
        with pytest.raises(ValueError):
            _class(data_sample, "sm", time_step = "invalid_time_step", normal_metrics=['mean'])
            
        # Test for invalid input parameter for the metrics   
        with pytest.raises(ValueError):
            _class(data_sample, "sm","month" ,normal_metrics = ["invalid_metric","another_invalid_metric"] )
            
            
    @pytest.mark.parametrize("time_steps, _class", [
        ("month", MonthlyAggregator),
        ("dekad", DekadalAggregator),
        ("week", WeeklyAggregator),
        ("bimonth", BimonthlyAggregator),
        ("day", DailyAggregator)
    ])
    def test_aggregate(self, climatology, time_steps, _class):
        """
        Test the aggregation method of Climatology class.
        """
        
        climatology.time_step = time_steps
        df = climatology.aggregate()
        assert isinstance(df, pd.DataFrame)
        assert f"{climatology.variable}-avg" in df.columns
        assert (df == _class(climatology.original_df, climatology.variable).aggregate()).all().all()    
            
        

    @pytest.mark.parametrize("time_step, metric, month, dekad, week, bimonth, day", 
        [
        ("month", 'mean', 2, None, None, None, None),
        ("month", 'median', 5, None, None, None, None),
        ("month", 'min', 8, None, None, None, None),
        ("month", 'max', 12, None, None, None, None),
        ("dekad", 'mean', 1, 1, None, None, None),
        ("dekad", 'median', 5, 2, None, None, None),
        ("dekad", 'min', 12, 3, None, None, None),
        ("dekad", 'max', 10, 1, None, None, None),
        ("week", 'mean', None, None, 10, None, None),
        ("week", 'median', None, None, 20, None, None),
        ("week", 'min', None, None, 30, None, None),
        ("week", 'max', None, None, 40, None, None),
        ("bimonth", 'mean', 12, None, None, 1, None),
        ("bimonth", 'median', 8, None, None, 2, None),
        ("bimonth", 'min', 4, None, None, 1, None),
        ("bimonth", 'max', 5, None, None, 2, None),
        ("day", 'mean', 2, None, None, None, 15),
        ("day", 'median', 7, None, None, None, 27),
        ("day", 'min', 12, None, None, None, 22),
        ("day", 'max', 4, None, None, None, 14)
        ])
    
    
    def test_climatology_compute_without_smoothing(self, clim_metrics_all, time_step, metric, month, dekad, week, bimonth, day):
        """
        Test the climate normals computed by Climatology class for different time steps,
              months, dekads, weeks, bimonths, and days with different metrics.
        """
        clim_metrics_all.time_step = time_step
        df = clim_metrics_all.compute_normals(month=month, dekad=dekad, week=week, bimonth=bimonth, day=day)
        
        expected_normal = df[f"{clim_metrics_all.variable}-avg"].agg(metric)
        computed_normal = random.choice(df[f'norm-{metric}'])
        assert computed_normal == pytest.approx(expected_normal, rel=1e-4)
        
    @pytest.mark.parametrize("time_step, metric, month, dekad, week, bimonth, day , window_size",
        [
        ("month", 'mean', 2, None, None, None, None, 5),
        ("month", 'median', 5, None, None, None, None,7),
        ("month", 'min', 8, None, None, None, None,3),
        ("month", 'max', 12, None, None, None, None,9),
        ("dekad", 'mean', 1, 1, None, None, None,11),
        ("dekad", 'median', 5, 2, None, None, None,15),
        ("dekad", 'min', 12, 3, None, None, None,13),
        ("dekad", 'max', 10, 1, None, None, None,15),
        ("week", 'mean', None, None, 10, None, None,17),
        ("week", 'median', None, None, 20, None, None,19),
        ("week", 'min', None, None, 30, None, None,21),
        ("week", 'max', None, None, 40, None, None,23),
        ("bimonth", 'mean', 12, None, None, 1, None,25),
        ("bimonth", 'median', 8, None, None, 2, None,27),
        ("bimonth", 'min', 4, None, None, 1, None,29),
        ("bimonth", 'max', 5, None, None, 2, None,31),
        ("day", 'mean', 2, None, None, None, 15,33),
        ("day", 'median', 7, None, None, None, 27,35),
        ("day", 'min', 12, None, None, None, 22,37),
        ("day", 'max', 4, None, None, None, 14,39)
        ])
    
    def test_climatology_compute_with_smoothing(self, clim_metrics_all, time_step, metric, month, dekad, week, bimonth, day, window_size):
        """
        Test the climate normals computed by Climatology class for different time steps,
              months, dekads, weeks, bimonths, and days with different metrics and smoothing.
        """
        clim_metrics_all.time_step = time_step
        clim_metrics_all.smoothing = True
        clim_metrics_all.smooth_window_size = window_size
        df = clim_metrics_all.compute_normals(month=month, dekad=dekad, week=week, bimonth=bimonth, day=day)
        
        expected_normal = df[f"{clim_metrics_all.variable}-avg"].agg(metric)
        computed_normal = random.choice(df[f'norm-{metric}'])
        assert computed_normal == pytest.approx(expected_normal, rel=1e-4)
       

            
    

    
   
    
if __name__ == "__main__":
    pytest.main([__file__])

