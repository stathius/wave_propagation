import os
from PIL import Image


"""
Wilhelm Sorteberg, 2018
wilhelm@sorteberg.eu


"""



Location = "./Video_Data"
Files = os.listdir(Location)
for File in Files:
    Images = os.listdir(Location + "/" + File)
    for Im in Images:
        try:
            Image.open(Location + "/" + File + "/" + Im)
        except:
            os.remove(Location + "/" + File + "/" + Im)
    Images = os.listdir(Location + "/" + File)
    if len(Images) != 100:
        for Im in Images:
            os.remove(Location + "/" + File + "/" + Im)
        os.removedirs(Location + "/" + File)
