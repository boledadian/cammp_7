import pandas as pd
import re

periodicTableDf = pd.read_csv('Periodic Table of Elements.csv')

trainDf = pd.read_csv('2017 high throughoutDFT calculations of formation energy stability and oxygen vacancy formation energy of ABO3 prerovskites.csv')

periodicADF = periodicTableDf.add_prefix("A-")
periodicBDF = periodicTableDf.add_prefix("B-")

periodicADF['A'] = periodicADF['A-Symbol']
periodicBDF['B'] = periodicBDF['B-Symbol']





trainDf = trainDf.merge(periodicADF,on='A')

trainDf = trainDf.merge(periodicBDF, on='B')
trainDf.to_csv('E_Vo_withDatafromPeriodicTable.csv',index=False)
