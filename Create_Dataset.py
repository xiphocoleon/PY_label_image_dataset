import numpy as np
import pandas as pd
import os

#print("Numpy version is: " + np.__version__)
#print("Pandas version is: " + pd.__version__)

# DATASET FORMAT AND LABELING 
# This script will output a .csv which will be one column of image paths
# And one column of labels
# The original data is stored as such:
# [Highest level path for entire dataset] --> as many individual folders of images, all of the same label
# The images will be labeled by the name of the folder that contains them
# E.g., [Highest level folder] / folder named "cats" with images. (the images in this folder will all be labeled "cats")


#Create dataframe (to become CSV) to put image data into:
df = pd.DataFrame()

#Create list to store all image names from all directories in first column of dataframe
complete_image_list = []
complete_label_list = []

#Read image file names in directory:
#Enter highest directory of images here:
path = 'C:\\Users\\thoma\\Pictures\\Geodis'
dir_list = os.listdir(path)
for x in range(len(dir_list)):
    #print("full paths of directories is: " + path + '\\' + dir_list[x])   
    subdirectory = path + '\\' + dir_list[x]
    subdirectory_list = os.listdir(subdirectory)

    #dir list should be the label for each set of images
    print(dir_list[x])
    
    for y in range(len(subdirectory_list)):
        #print("full paths of images is: " + path + '\\' + dir_list[x] + '\\' + subdirectory_list[y])
        image_path = path + '\\' + dir_list[x] + '\\' + subdirectory_list[y]
        image_label = dir_list[x]
        
        #Add image paths to a list
        complete_image_list.append(image_path)
        
        #Add image labels to a list
        complete_label_list.append(image_label)

        
print('The size of the full image list is: ' + str(len(complete_image_list)))

#Filename is the column head for the first column
df.insert(0, 'filename', complete_image_list)

#Label list is the column header for the second column
df.insert(1, 'label', complete_label_list)
print(df)

#Convert dataframe to csv with no column headers included or row numbers:
df.to_csv('C:\\Users\\thoma\\Documents\\Computer Vision\\geodis.csv', sep=',', na_rep='NaN', header=1, index=0)