
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from itertools import combinations 
import csv
#import pandas


# load in 15 images
# abbreviationsof: bobolark, hooded warbler, indigo bunting, marsh wren, painted bunting, shiny cowbird
images = ["bl1", "bl2", "bl3","hw1","hw2", "hw3", "ib1","ib2", "ib3","mw1", "mw2", "mw3", "pb1", "pb2", "pb3", "sc1", "sc2", "sc3"]

# figure size in inches optional
rcParams['figure.figsize'] = 11 ,8
    
# read images
combos = list(combinations(images, 2)) #all the combinations, n choose 2
print(combos)

## Prints list of animals in correct order in a CSV file titled "file"
#df = pandas.DataFrame(data={"col1": combos})
#df.to_csv("./file.csv", sep=',',index=False)

# Plots the images in pairs and saves them in a folder.
"""
def plot_images(img_a, img_b):
# displays images a and b
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img_a);
    ax[0].axis('off')
    ax[1].imshow(img_b);
    ax[1].axis('off')
    return

for i in range(0,len(combos)):
    img_a = mpimg.imread(combos[i][0]+".jpg")
    img_b = mpimg.imread(combos[i][1]+".jpg")
    plot_images(img_a, img_b)
    plt.savefig("ImagePairs/" + combos[i][0] + "_" + combos[i][1] + ".jpg", bbox_inches='tight')
"""