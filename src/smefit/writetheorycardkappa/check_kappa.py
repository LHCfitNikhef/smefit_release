#Sum of kappa in a dataset must be 1 for each element
import json
import sys
import numpy as np
dataset=str(sys.argv[1])
with open(dataset, 'r') as file:
        data = json.load(file)

keys=data["LO"].keys()

n=len(list(data["LO"][list(keys)[0]]))
arr=np.zeros(n)
for k in keys:
        arr+=np.array(data["LO"][k])
print(arr)
        
        