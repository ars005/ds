
import pandas as pd
import numpy as np

data={
    "ID":["E101","E102","E103","E104","E105"],
    "Name":["Atul","Raj","Darpan","Anmol","Piyush"],
    "Designation":["Manager","clerk","analyst","manager","clerk"],
    "Salary":[56000,25000,35000,28000,58000],
    "Bonus":[15000,7000,9000,13000,12000]
}
# print(data)
df=pd.DataFrame(data)
print(df)