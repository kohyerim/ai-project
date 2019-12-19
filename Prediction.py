from mllib.prediction_util import PredictionUtil
import pandas as pd

util = PredictionUtil()
util.read('dataset.csv')
print(util.df.columns)
util.drop(['Unnamed: 0', 'brand_name', 'Location', 'New_Price'])
util.boxplot('Owner_Type', 'Price')
util.lmplot('Power_upd', 'Price', 'Owner_Type')
util.df = pd.get_dummies(util.df)
print(util.df.columns)
util.heatmap(['Kilometers_Driven', 'Seats', 'Price', 'Mileage_upd', 'Engine_upd',
               'Power_upd', 'Year_upd', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
               'Fuel_Type_Electric', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
               'Transmission_Automatic', 'Transmission_Manual', 'Owner_Type_First',
               'Owner_Type_Fourth & Above', 'Owner_Type_Second', 'Owner_Type_Third'])

util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual'], 'Price')
print('\n# add Year')
util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual', 'Year_upd'], 'Price')
print('\n# add Mileage')
util.run_all(['Power_upd', 'Engine_upd', 'Transmission_Manual', 'Year_upd', 'Mileage_upd'], 'Price')