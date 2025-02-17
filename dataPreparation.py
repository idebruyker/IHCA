import pandas as pd

# 53904_1

pd1tcfcells_53904_1 = pd.read_csv('data/53904_1/53904_1pd1tcfcells.csv')
pd1cells_53904_1 = pd.read_csv('data/53904_1/53904_1pd1cells.csv')
tcfcells_53904_1 = pd.read_csv('data/53904_1/53904_1tcfcells.csv')
cd8cells_53904_1 = pd.read_csv('data/53904_1/53904_1cd8cells.csv')
cd4cells_53904_1 = pd.read_csv('data/53904_1/53904_1cd4cells.csv')
mhccells_53904_1 = pd.read_csv('data/53904_1/53904_1mhccells.csv')

print('Original 53904_1:')
print('pd1tfc',pd1tcfcells_53904_1.shape[0])
print('pd1',pd1cells_53904_1.shape[0])
print('tcf',tcfcells_53904_1.shape[0])
print('cd8',cd8cells_53904_1.shape[0])
print('cd4',cd4cells_53904_1.shape[0])
print('mhc',mhccells_53904_1.shape[0])

# remove rows identified as pd1tcf from pd1 based upon Object.ID
pd1cells_53904_1 = pd1cells_53904_1[~pd1cells_53904_1['Object.ID'].isin(pd1tcfcells_53904_1['Object.ID'])]
# remove rows identified as pd1tcf from cd8 based upon Object.ID
cd8cells_53904_1 = cd8cells_53904_1[~cd8cells_53904_1['Object.ID'].isin(pd1tcfcells_53904_1['Object.ID'])]
# remove rows identified as pd1 from cd8 based upon Object.ID
cd8cells_53904_1 = cd8cells_53904_1[~cd8cells_53904_1['Object.ID'].isin(pd1cells_53904_1['Object.ID'])]

print('Updated 53904_1:')
print('pd1tfc',pd1tcfcells_53904_1.shape[0])
print('pd1',pd1cells_53904_1.shape[0])
print('tcf',tcfcells_53904_1.shape[0])
print('cd8',cd8cells_53904_1.shape[0])
print('cd4',cd4cells_53904_1.shape[0])
print('mhc',mhccells_53904_1.shape[0])

pd1tcfcells_53904_1.to_csv('data/53904_1/53904_1pd1tcfcells.csv',index=False)
pd1cells_53904_1.to_csv('data/53904_1/53904_1pd1cells.csv',index=False)
tcfcells_53904_1.to_csv('data/53904_1/53904_1tcfcells.csv',index=False)
cd8cells_53904_1.to_csv('data/53904_1/53904_1cd8cells.csv',index=False)
cd4cells_53904_1.to_csv('data/53904_1/53904_1cd4cells.csv',index=False)
mhccells_53904_1.to_csv('data/53904_1/53904_1mhccells.csv',index=False)

# 33683_1

pd1tcfcells_33683_1 = pd.read_csv('data/33683_1/33683_1pd1tcfcells.csv')
pd1cells_33683_1 = pd.read_csv('data/33683_1/33683_1pd1cells.csv')
tcfcells_33683_1 = pd.read_csv('data/33683_1/33683_1tcfcells.csv')
cd8cells_33683_1 = pd.read_csv('data/33683_1/33683_1cd8cells.csv')
cd4cells_33683_1 = pd.read_csv('data/33683_1/33683_1cd4cells.csv')
mhccells_33683_1 = pd.read_csv('data/33683_1/33683_1mhccells.csv')

print('Original 33683_1:')
print('pd1tfc',pd1tcfcells_33683_1.shape[0])
print('pd1',pd1cells_33683_1.shape[0])
print('tcf',tcfcells_33683_1.shape[0])
print('cd8',cd8cells_33683_1.shape[0])
print('cd4',cd4cells_33683_1.shape[0])
print('mhc',mhccells_33683_1.shape[0])

# remove rows identified as pd1tcf from pd1 based upon Object.ID
pd1cells_33683_1 = pd1cells_33683_1[~pd1cells_33683_1['Object.ID'].isin(pd1tcfcells_33683_1['Object.ID'])]
# remove rows identified as pd1tcf from cd8 based upon Object.ID
cd8cells_33683_1 = cd8cells_33683_1[~cd8cells_33683_1['Object.ID'].isin(pd1tcfcells_33683_1['Object.ID'])]
# remove rows identified as pd1 from cd8 based upon Object.ID
cd8cells_33683_1 = cd8cells_33683_1[~cd8cells_33683_1['Object.ID'].isin(pd1cells_33683_1['Object.ID'])]

print('Updated 33683_1:')
print('pd1tfc',pd1tcfcells_33683_1.shape[0])
print('pd1',pd1cells_33683_1.shape[0])
print('tcf',tcfcells_33683_1.shape[0])
print('cd8',cd8cells_33683_1.shape[0])
print('cd4',cd4cells_33683_1.shape[0])
print('mhc',mhccells_33683_1.shape[0])

pd1tcfcells_33683_1.to_csv('data/33683_1/33683_1pd1tcfcells.csv',index=False)
pd1cells_33683_1.to_csv('data/33683_1/33683_1pd1cells.csv',index=False)
tcfcells_33683_1.to_csv('data/33683_1/33683_1tcfcells.csv',index=False)
cd8cells_33683_1.to_csv('data/33683_1/33683_1cd8cells.csv',index=False)
cd4cells_33683_1.to_csv('data/33683_1/33683_1cd4cells.csv',index=False)
mhccells_33683_1.to_csv('data/33683_1/33683_1mhccells.csv',index=False)

# 33270_2

pd1tcfcells_33270_2 = pd.read_csv('data/33270_2/33270_2pd1tcfcells.csv')
pd1cells_33270_2 = pd.read_csv('data/33270_2/33270_2pd1cells.csv')
tcfcells_33270_2 = pd.read_csv('data/33270_2/33270_2tcfcells.csv')
cd8cells_33270_2 = pd.read_csv('data/33270_2/33270_2cd8cells.csv')
cd4cells_33270_2 = pd.read_csv('data/33270_2/33270_2cd4cells.csv')
mhccells_33270_2 = pd.read_csv('data/33270_2/33270_2mhccells.csv')

print('Original 33270_2:')
print('pd1tfc',pd1tcfcells_33270_2.shape[0])
print('pd1',pd1cells_33270_2.shape[0])
print('tcf',tcfcells_33270_2.shape[0])
print('cd8',cd8cells_33270_2.shape[0])
print('cd4',cd4cells_33270_2.shape[0])
print('mhc',mhccells_33270_2.shape[0])

# remove rows identified as pd1tcf from pd1 based upon Object.ID
pd1cells_33270_2 = pd1cells_33270_2[~pd1cells_33270_2['Object.ID'].isin(pd1tcfcells_33270_2['Object.ID'])]
# remove rows identified as pd1tcf from cd8 based upon Object.ID
cd8cells_33270_2 = cd8cells_33270_2[~cd8cells_33270_2['Object.ID'].isin(pd1tcfcells_33270_2['Object.ID'])]
# remove rows identified as pd1 from cd8 based upon Object.ID
cd8cells_33270_2 = cd8cells_33270_2[~cd8cells_33270_2['Object.ID'].isin(pd1cells_33270_2['Object.ID'])]

print('Updated 33270_2:')
print('pd1tfc',pd1tcfcells_33270_2.shape[0])
print('pd1',pd1cells_33270_2.shape[0])
print('tcf',tcfcells_33270_2.shape[0])
print('cd8',cd8cells_33270_2.shape[0])
print('cd4',cd4cells_33270_2.shape[0])
print('mhc',mhccells_33270_2.shape[0])

pd1tcfcells_33270_2.to_csv('data/33270_2/33270_2pd1tcfcells.csv',index=False)
pd1cells_33270_2.to_csv('data/33270_2/33270_2pd1cells.csv',index=False)
tcfcells_33270_2.to_csv('data/33270_2/33270_2tcfcells.csv',index=False)
cd8cells_33270_2.to_csv('data/33270_2/33270_2cd8cells.csv',index=False)
cd4cells_33270_2.to_csv('data/33270_2/33270_2cd4cells.csv',index=False)
mhccells_33270_2.to_csv('data/33270_2/33270_2mhccells.csv',index=False)