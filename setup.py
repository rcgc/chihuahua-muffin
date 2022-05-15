""""
    - Will rename files in folders
    - Will generate train, validation and test datasets
"""
import os, shutil, pathlib


def rename_images(folder_name, image_prefix, image_extension):
    folder = r'./'+folder_name+'/'
    count = 1
    # count increase by 1 in each iteration
    # iterate all files from a directory
    for file_name in os.listdir(folder):
        # Construct old file name
        source = folder + file_name

        # Adding the count to the new file name and extension
        destination = folder + image_prefix + str(count) + image_extension

        # Renaming the file
        if os.path.isfile(destination):
            print("The file " + destination + " already exists")
        else:
            os.rename(source, destination)
            count += 1

    print('All Files Renamed')

    print('New Names are')
    # verify the result
    res = os.listdir(folder)
    print(res)


# Rename corresponding folders
# rename_images("Muffins", "muffin_", ".JPG")
# rename_images("Chihuahuas", "chihuahua_", ".JPG")

# Create training, validation and test dataset
