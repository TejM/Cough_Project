
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# with h5py.File('/Users/tejasmallela/Desktop/CS 3990 - Research - Cough Detection/coughproject/src/f1/features.hdf5', 'r') as hdf:
#     ls = list(hdf.items())
#     print (ls)
#     data = hdf.get('key')
#     g1 = list(data.items())
#     print (g1)
#     dataset = (data.get('table'))
#     print(dataset[:10])


# Paths for features and annotations HDF5 Files
features = '/Users/tejasmallela/Desktop/CS 3990 - Research - Cough Detection/coughproject/src/f1/features.hdf5'
annotations = '/Users/tejasmallela/Desktop/CS 3990 - Research - Cough Detection/coughproject/src/f1/annotations.hdf5'

# Load the dataframes using pandas
df = pd.read_hdf(features)
df1 = pd.read_hdf(annotations)
# print(df.head())

# Adding the coverage values to the features dataframe
df['coverage'] = df1['coverage']
del df1
# print(df.shape)

# Consider 95% coverage of cough window as a cough value
df['cough'] = np.where(df['coverage'] >= 95, 1, 0)
# print(df['cough'].value_counts())

# Separate independent and dependent variables
x = df.iloc[:, 0:30]
# print(x.head)
y = df['cough']
# print(y.head)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# print(x_train.shape)
# print(x_test.shape)
#
# print(x_train.head)
# print(x_test.head)
#
# print(y_train.shape)
# print(y_test.shape)
#
# print(y_train.head)
# print(y_test.head)

# TODO apply Cross validation later
model = RandomForestClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
