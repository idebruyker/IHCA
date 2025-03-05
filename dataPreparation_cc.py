import pandas as pd

# CC01_PreNivo

pd1tcfcells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivopd1tcfcells.csv')
pd1cells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivopd1cells.csv')
tcfcells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivotcfcells.csv')
cd8cells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivocd8cells.csv')
cd4cells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivocd4cells.csv')
mhccells_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivomhccells.csv')

print('Original CC01_PreNivo:')
print('pd1tfc',pd1tcfcells_CC01_PreNivo.shape[0])
print('pd1',pd1cells_CC01_PreNivo.shape[0])
print('tcf',tcfcells_CC01_PreNivo.shape[0])
print('cd8',cd8cells_CC01_PreNivo.shape[0])
print('cd4',cd4cells_CC01_PreNivo.shape[0])
print('mhc',mhccells_CC01_PreNivo.shape[0])

# remove rows identified as pd1tcf from pd1 based upon Object.ID
pd1cells_CC01_PreNivo = pd1cells_CC01_PreNivo[~pd1cells_CC01_PreNivo['Object.ID'].isin(pd1tcfcells_CC01_PreNivo['Object.ID'])]
# remove rows identified as pd1tcf from cd8 based upon Object.ID
cd8cells_CC01_PreNivo = cd8cells_CC01_PreNivo[~cd8cells_CC01_PreNivo['Object.ID'].isin(pd1tcfcells_CC01_PreNivo['Object.ID'])]
# remove rows identified as pd1 from cd8 based upon Object.ID
cd8cells_CC01_PreNivo = cd8cells_CC01_PreNivo[~cd8cells_CC01_PreNivo['Object.ID'].isin(pd1cells_CC01_PreNivo['Object.ID'])]

print('Updated CC01_PreNivo:')
print('pd1tfc',pd1tcfcells_CC01_PreNivo.shape[0])
print('pd1',pd1cells_CC01_PreNivo.shape[0])
print('tcf',tcfcells_CC01_PreNivo.shape[0])
print('cd8',cd8cells_CC01_PreNivo.shape[0])
print('cd4',cd4cells_CC01_PreNivo.shape[0])
print('mhc',mhccells_CC01_PreNivo.shape[0])

pd1tcfcells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivopd1tcfcells.csv',index=False)
pd1cells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivopd1cells.csv',index=False)
tcfcells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivotcfcells.csv',index=False)
cd8cells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivocd8cells.csv',index=False)
cd4cells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivocd4cells.csv',index=False)
mhccells_CC01_PreNivo.to_csv('data/CC01_PreNivo/CC01_PreNivomhccells.csv',index=False)

# CC09_PreNivo

pd1tcfcells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivopd1tcfcells.csv')
pd1cells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivopd1cells.csv')
tcfcells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivotcfcells.csv')
cd8cells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivocd8cells.csv')
cd4cells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivocd4cells.csv')
mhccells_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivomhccells.csv')

print('Original CC09_PreNivo:')
print('pd1tfc',pd1tcfcells_CC09_PreNivo.shape[0])
print('pd1',pd1cells_CC09_PreNivo.shape[0])
print('tcf',tcfcells_CC09_PreNivo.shape[0])
print('cd8',cd8cells_CC09_PreNivo.shape[0])
print('cd4',cd4cells_CC09_PreNivo.shape[0])
print('mhc',mhccells_CC09_PreNivo.shape[0])

# remove rows identified as pd1tcf from pd1 based upon Object.ID
pd1cells_CC09_PreNivo = pd1cells_CC09_PreNivo[~pd1cells_CC09_PreNivo['Object.ID'].isin(pd1tcfcells_CC09_PreNivo['Object.ID'])]
# remove rows identified as pd1tcf from cd8 based upon Object.ID
cd8cells_CC09_PreNivo = cd8cells_CC09_PreNivo[~cd8cells_CC09_PreNivo['Object.ID'].isin(pd1tcfcells_CC09_PreNivo['Object.ID'])]
# remove rows identified as pd1 from cd8 based upon Object.ID
cd8cells_CC09_PreNivo = cd8cells_CC09_PreNivo[~cd8cells_CC09_PreNivo['Object.ID'].isin(pd1cells_CC09_PreNivo['Object.ID'])]

print('Updated CC09_PreNivo:')
print('pd1tfc',pd1tcfcells_CC09_PreNivo.shape[0])
print('pd1',pd1cells_CC09_PreNivo.shape[0])
print('tcf',tcfcells_CC09_PreNivo.shape[0])
print('cd8',cd8cells_CC09_PreNivo.shape[0])
print('cd4',cd4cells_CC09_PreNivo.shape[0])
print('mhc',mhccells_CC09_PreNivo.shape[0])

pd1tcfcells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivopd1tcfcells.csv',index=False)
pd1cells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivopd1cells.csv',index=False)
tcfcells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivotcfcells.csv',index=False)
cd8cells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivocd8cells.csv',index=False)
cd4cells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivocd4cells.csv',index=False)
mhccells_CC09_PreNivo.to_csv('data/CC09_PreNivo/CC09_PreNivomhccells.csv',index=False)
