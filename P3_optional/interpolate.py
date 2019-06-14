

import pandas as pd
import re

valDf = pd.read_excel('E_Vo_dopedPerovskites_extended.xlsx')

trainDf = pd.read_csv('E_Vo_withDatafromPeriodicTable.csv')


print(valDf)

features=['Radius A [ang]','Radius B [ang]',"Volume per atom [A^3/atom]","Formation energy [eV/atom]","Stability [eV/atom]","Magnetic moment [mu_B]","Band gap [eV]","a [ang]","b [ang]","c [ang]","alpha [deg]","beta [deg]","gamma [deg]",'A-AtomicNumber','A-AtomicMass','A-NumberofNeutrons','A-NumberofProtons','A-NumberofElectrons','A-Period','A-Group','A-AtomicRadius','A-Electronegativity','A-FirstIonization','A-Density','A-MeltingPoint','A-BoilingPoint','A-NumberOfIsotopes','A-Year','A-SpecificHeat','A-NumberofShells','A-NumberofValence','B-AtomicNumber','B-AtomicMass','B-NumberofNeutrons','B-NumberofProtons','B-NumberofElectrons','B-Period','B-Group','B-AtomicRadius','B-Electronegativity','B-FirstIonization','B-Density','B-MeltingPoint','B-BoilingPoint','B-NumberOfIsotopes','B-Year','B-SpecificHeat','B-NumberofShells','B-NumberofValence']



for s in features:
    valDf[s] = valDf.apply(lambda row: float(trainDf.loc[trainDf['A'] == row.A1].iloc[0][s]) * row.PercentA1 + float(trainDf.loc[trainDf['A'] == row.A2].iloc[0][s]) * (1- row.PercentA1),axis=1)





valDf.to_csv('E_Vo_dopedPerovskites_interpolated.csv',index=False)