from unittest.mock import Mock
import pytest
import pandas as pd
from ssmad.climatology import Aggregator, MonthlyAggregator,\
                              DekadalAggregator, WeeklyAggregator,\
                              BimonthlyAggregator, DailyAggregator\
                              


class TestAggregator:
    """
    Class for testing the Aggregator class.
    """
           
    @pytest.fixture
    def aggregator(self, data_sample):
        """
        Fixture to create an instance of Aggregator for testing.
        """
        return Aggregator(df = data_sample, variable="sm", mode='subset',\
            smoothing=False, window_size=None, start_date='2022-01-01', end_date='2022-12-31')
        
    def test_initialization(self, aggregator, data_sample):
        """
        Test the initialization of the Aggregator class.
        """
        assert aggregator is not None
        assert aggregator.original_df.equals(data_sample)
        assert aggregator.variable == "sm"
        assert aggregator.mode == "subset"
        assert aggregator.start_date == '2022-01-01'
        assert aggregator.end_date == '2022-12-31'
        
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
            
        
    def test_mode_subset(self, data_sample , _class = Aggregator):
        """
        Test subset mode of Aggregator class.
        """
        agg = _class(data_sample, 'sm',  mode='subset', start_date='2022-01-01', end_date='2022-01-02')
        
        # Check if the length of the aggregated DataFrame is less than the original DataFrame
        assert len(agg.df) != len(data_sample)  
        
        # Check if the length of the aggregated DataFrame is equal to the number of days in the subset
        assert len(agg.df) == 2  
        
    def test_mode_subset_no_dates(self, data_sample , _class = Aggregator):
        """
        Test subset mode of Aggregator class without dates.
        """
        
        # Test for invalid input parameter for the start_date and end_date
        with pytest.raises(TypeError):
            _class(data_sample, 'sm', mode='subset', start_date='2022-01-01')
        with pytest.raises(TypeError):
            _class(data_sample, 'sm', mode='subset', end_date='2022-01-01')
        with pytest.raises(TypeError):
            _class(data_sample, 'sm', mode='subset', start_date='wrong_format', end_date='wrong_format')
            
            
    def test_mode_all(self, data_sample, _class = Aggregator):
        """
        Test all mode of Aggregator class.
        """
        agg = _class(data_sample, 'sm', mode='all')
        # Check if the length of the aggregated DataFrame is equal to the original DataFrame
        assert len(agg.original_df) == len(data_sample)  
        
    def test_smoothing(self, data_sample, _class = Aggregator , variable = "sm"):
        """
        Test smoothing of Aggregator class.
        """
        df_without_smoothing = _class(data_sample, "sm", smoothing=False)\
            .aggregate(start_date='2022-01-17', end_date='2022-01-26')
        df_with_smoothing = _class(data_sample, "sm", smoothing=True, window_size=9)\
            .aggregate(year = 2022, month = 1, day = 21)
        
        # Check if the new value of the variable is the mean of the original values within the window
        assert df_without_smoothing[variable].mean() == pytest.approx(df_with_smoothing[variable].values[0])
        
        
    
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
        return MonthlyAggregator(df = data_sample, variable="sm",mode='subset'\
            ,smoothing=False, window_size=None, start_date='2022-01-01', end_date='2022-12-31')
    
    
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
        return DekadalAggregator(df = data_sample, variable="sm", smoothing=False\
            , mode='subset', start_date='2022-01-01', end_date='2022-12-31')
    
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
        
        # Check if the class adds a new column ['dekad] to the aggregated DataFrame
        assert f"dekad" in  df.columns
        
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
        return WeeklyAggregator(df = data_sample, variable="sm", smoothing=False\
            , mode='subset', start_date='2022-01-01', end_date='2022-12-31')
    
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
        return BimonthlyAggregator(df = data_sample, variable="sm", smoothing=False\
            , mode='subset', start_date='2022-01-01', end_date='2022-12-31')
    
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
        return DailyAggregator(df = data_sample, variable="sm", smoothing=False\
            , mode='subset', start_date='2022-01-01', end_date='2022-12-31')
        
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

        
    def test_drop_duplicates(self , aggregator, variable = "sm"):
        """
        Test the drop_duplicates of DailyAggregator class.
        """
        super().test_drop_duplicates(aggregator)
    
    def test_aggregate_filter_df(data_sample , aggregator):
        """
        Test the aggregation method of DailyAggregator class with filtering.
        """
        
        resulted_df = aggregator.aggregate(year = 2022, month = 1, day = 1)
        # Check if aggregation result has 1 row (January 1, 2022)
        assert len(resulted_df) == 1
    
        
        
        
    
   
    
if __name__ == "__main__":
    pytest.main([__file__])

