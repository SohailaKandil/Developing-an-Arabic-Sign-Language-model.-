from matplotlib.image import imread
import csv
import numpy as np
#from skimage import  transform

test_data =  open ("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test.csv" , mode = "w" , newline="")
writer = csv.writer(test_data)
writer.writerow(["label"]  + [f"pixel{x}" for x in range (4096)])
   
new_size = (64,64)    

file_names = open ("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/ArSL_Data_Labels.csv" , mode ="r")
reader = csv.DictReader(file_names)

num = 0
for row in reader: 
    try:
        num+=1
        print(num)
        file_name = row["File_Name"]
        fclass = row["Class"]
        print(file_name)
        print(fclass)
        image = "C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/ArASL_Database_54K_Final" + "/" + fclass + "/" + file_name
        image = imread(image)
      
        rescaled_image = np.copy(image)
        rescaled_image = image.resize(new_size)
        rescaled_image = image.flatten()
        print(rescaled_image)
        writer.writerow([fclass] + [x for x in rescaled_image] )
    
    except:
        continue

test_data.close()
file_names.close()
    