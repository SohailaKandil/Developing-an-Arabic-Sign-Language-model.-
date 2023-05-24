import pandas as pd

weather_data_path = "C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/weather data/seattle-weather.csv"
data = pd.read_csv(weather_data_path ,index_col = 0)
data = data.ffill()
print(data.head(5))

data.plot.scatter("temp_max" , "temp_min")
