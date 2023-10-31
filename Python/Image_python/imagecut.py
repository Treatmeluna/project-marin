from rembg import remove
from PIL import Image
import glob

file_path1="c:/ocean/Dataset/Porpoise/"
file_path2="c:/ocean/Dataset/Mink/"


result_path1="c:/ocean/result/Porpoise/"
result_path2="c:/ocean/result/Mink/"

list_images1 = glob.glob(file_path1+"*.jpg")
list_images2 = glob.glob(file_path2+"*.jpg")

i = 0

for image in list_images1:
    input = Image.open(image) # load image
    output = remove(input) # remove background
    output.save(result_path1 + f'{i}.png') # save image
    i +=1 
    
    
for image in list_images2:
    input = Image.open(image) # load image
    output = remove(input) # remove background
    output.save(result_path2 + f'{i}.png') # save image
    i +=1 
    

    

    
    