# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
data['Description']= label_encoder.fit_transform(data['Description'])
data['Transaction Type']= label_encoder.fit_transform(data['Transaction Type'])
data['Category']= label_encoder.fit_transform(data['Category'])
data['Month']= label_encoder.fit_transform(data['Month'])
data['Year']= label_encoder.fit_transform(data['Year']) 
