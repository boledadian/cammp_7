

import pandas as pd
import re

valDf = pd.read_excel('E_Vo_dopedPerovskites_extended.xlsx')

trainDf = pd.read_csv('2017 high throughoutDFT calculations of formation energy stability and oxygen vacancy formation energy of ABO3 prerovskites.csv')


print(valDf)

features=['Radius A [ang]','Radius B [ang]',"Volume per atom [A^3/atom]","Formation energy [eV/atom]","Stability [eV/atom]","Magnetic moment [mu_B]","Band gap [eV]","a [ang]","b [ang]","c [ang]","alpha [deg]","beta [deg]","gamma [deg]"]



for s in features:
    valDf[s] = valDf.apply(lambda row: float(trainDf.loc[trainDf['A'] == row.A1].iloc[0][s]) * row.PercentA1 + float(trainDf.loc[trainDf['A'] == row.A2].iloc[0][s]) * (1- row.PercentA1),axis=1)



print(valDf)

valDf.to_csv('E_Vo_dopedPerovskites_interpolated.csv',index=False)