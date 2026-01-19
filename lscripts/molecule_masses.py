from molmass import Formula
import numpy as np
import pandas as pd

# read in molecule names
df = pd.read_csv('../data/molecules.txt', sep='\t')
# calculate masses
mass_arr = np.zeros(len(df.iloc[:,0]))
for i in range(len(df.iloc[:,0])):
    molecule = df.iloc[i,0]
    mass_arr[i] = Formula(molecule).mass
# show masses
mass_dict = pd.DataFrame({
    'Molecule': df.iloc[:,0],
    'Mass': mass_arr
})
print(mass_dict)

