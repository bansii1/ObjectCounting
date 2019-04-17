

folder = "/home/bansi/tensoflow/11032018/n"

import os

i=0
for root, dirs, filenames in os.walk(folder):


    for filename in filenames:  
        fullpath = os.path.join(root, filename)  
        filename_split = os.path.splitext(filename) # filename will be filename_split[0] and extension will be filename_split[1])
        print (fullpath)
        print (filename_split[0])
        print (filename_split[1])
        os.rename(os.path.join(root, filename), os.path.join(root, "p" + str(i) + filename_split[1]))
        i=i+1
