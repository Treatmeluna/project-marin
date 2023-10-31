from rembg import remove
from PIL import Image

input = Image.open('tt.jpg') # load image
output = remove(input) # remove background
output.save('tt2.PNG') # save image