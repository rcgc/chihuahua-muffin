"""
    - Will rename files in folders
    - Will generate train, validation and test datasets
"""
import os, shutil, pathlib


# renames instances
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


# creates subsets
def make_subset(category, subset_name, start_index, end_index):
    original_dir = pathlib.Path(category)
    new_base_dir = pathlib.Path("chihuahua_vs_muffin")

    dir = new_base_dir / subset_name / category
    os.makedirs(dir)
    fnames = [f"{category}_{i}.JPG" for i in range(start_index, end_index)]
    for fname in fnames:
        shutil.copyfile(src=original_dir / fname, dst=dir / fname)


# Folder name, imageName, imageFormat
#rename_images("muffin", "muffin_", ".JPG")
#rename_images("chihuahua", "chihuahua_", ".JPG")

# Create training, validation and test dataset
make_subset("chihuahua", "train", 1, 251)
make_subset("chihuahua", "validation", 251, 401)
make_subset("chihuahua", "test", 401, 501)
make_subset("muffin", "train", 1, 251)
make_subset("muffin", "validation", 251, 401)
make_subset("muffin", "test", 401, 501)
