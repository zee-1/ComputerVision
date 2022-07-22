from re import L
from traceback import print_tb
from PIL import Image
from PIL import ImageOps

from os.path import exists
import os

from platform import uname

deleteComd ={"Windows":"del ","Linux":"rm "}
allowed_extentions = {"jpeg","jpg","png"}
file_path = input("Enter file path:")
if exists(file_path):
    path_list=file_path.split('\\')
    extention = path_list[-1].split('.')[-1]
    if extention in allowed_extentions:
        img = Image.open(file_path)
        img_gray = ImageOps.grayscale(img)
        img_gray.mode:L
        saveAs = "grayed_"+path_list[-1]
        img_gray.save(saveAs)
        preview = Image.open(saveAs)
        preview.show()
        print("You want to save the image")
        save = input("[Y]es/[N]o\n")
        if save=='Y' or save =='N':
            if save=='N':
                os.system(deleteComd[uname()[0]]+saveAs)
            else:
                print("File successfully saved")
        else:
            print("Invalid input! Saving file....")
    else:
        print("Invalid file format. Try using a jpg,jpeg or a png file.")
            

else:
    print("Invalid Path entered Please try again.")
