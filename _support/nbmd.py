import sys
import os
SUMMARYSTUFF = """
## Contents
{:.no_toc}
*  
{: toc}
"""
filetoread = sys.argv[1]
fdtoread = open(filetoread)
fileprefix = ".".join(filetoread.split('.')[:-1])
filetowrite = fileprefix+".newmd"
buffer = ""
firstheader = True
for line in fdtoread:
    if firstheader and line[0:2]=='# ':#assume title
        title = line.strip()[2:]
        firstheader = False
    else:
        buffer = buffer + line
fdtoread.close()
preamble = "title: {}\nnotebook: {}\n".format(title, fileprefix+".ipynb" )
preamble = "---\n"+preamble+"---\n"
fdtowrite=open(filetowrite, "w")
summarystuff = SUMMARYSTUFF
fdtowrite.write(preamble+summarystuff+buffer)
fdtowrite.close()
os.rename(filetowrite, filetoread)
