import csv
import pandas as pd

test_data =  open ("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test.csv" , mode = "r")
reader = csv.DictReader(test_data)
   
   

df = pd.read_csv("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test.csv", encoding='Windows-1256')
df = df.dropna()
df.to_csv("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test_cleaned.csv" ) 
test_data.close()