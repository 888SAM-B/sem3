import numpy as np
import pandas as pd
import random
dataset={
    "Emp_id":["Emp0"+str(i) for i in range(1,501)],
    "salary":[random.randint(10000,100000) for i in range(500)]

         }
d=pd.DataFrame(dataset)
d.to_csv('DataSet.csv',index=False)
print(d["salary"].mean())
print(d["salary"].median())
print(d["salary"].var())
print(d["salary"].std())
print(d["salary"].mode()[0])
