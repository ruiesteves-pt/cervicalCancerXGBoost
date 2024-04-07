# Cervical Cancer Prediction using XGBoost Model
# @ruiesteves-pt

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report



# Import and parse data as dataframe
cancerDf = pd.read_csv('cervical_cancer.csv')
# Composed by: 
# (int) Age
# (int) Number of sexual partners
#  (int) First sexual intercourse (age)
# (int) Num of pregnancies
# (bool) Smokes
# (bool) Smokes (years)
# (bool) Smokes (packs/year)
# (bool) Hormonal Contraceptives
# (int) Hormonal Contraceptives (years)
# (bool) IUD ("IUD" stands for "intrauterine device" and used for birth control
# (int) IUD (years)
# (bool) STDs (Sexually transmitted disease)
# (int) STDs (number)
# (bool) STDs:condylomatosis
# (bool) STDs:cervical condylomatosis
# (bool) STDs:vaginal condylomatosis
# (bool) STDs:vulvo-perineal condylomatosis
# (bool) STDs:syphilis
# (bool) STDs:pelvic inflammatory disease
# (bool) STDs:genital herpes
# (bool) STDs:molluscum contagiosum
# (bool) STDs:AIDS
# (bool) STDs:HIV
# (bool) STDs:Hepatitis B
# (bool) STDs:HPV
# (int) STDs: Number of diagnosis
# (int) STDs: Time since first diagnosis
# (int) STDs: Time since last diagnosis
# (bool) Dx:Cancer
# (bool) Dx:CIN
# (bool) Dx:HPV
# (bool) Dx
# (bool) Hinselmann: target variable - A colposcopy is a procedure in which doctors examine the cervix. 
# (bool) Schiller: target variable - Schiller's Iodine test is used for cervical cancer diagnosis
# (bool) Cytology: target variable - Cytology is the exam of a single cell type used for cancer screening.
# (bool) Biopsy: target variable - Biopsy is performed by removing a piece of tissue and examine it under microscope, 
# Biopsy is the main way doctors diagnose most types of cancer. 

print(cancerDf.describe())

# Process data and fill bad values with NaN
cancerDf = cancerDf.replace('?',np.nan)
plt.figure(figsize = (20,20))
sns.heatmap(cancerDf.isnull(),yticklabels = False)
plt.show()

# Remove mostly null-columns
cancerDf = cancerDf.drop(columns = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
cancerDf = cancerDf.apply(pd.to_numeric)
print(cancerDf.info())

# Replace null values with mean
cancerDf = cancerDf.fillna(cancerDf.mean())

# Nan heatmap
plt.figure(figsize = (20,20))
sns.heatmap(cancerDf.isnull(),yticklabels = False)
plt.show()

# Get the correlation matrix
corr_matrix = cancerDf.corr()
print(corr_matrix)

# Plot the correlation matrix
plt.figure(figsize = (30,30))
sns.heatmap(corr_matrix, annot = True)
plt.show()

# Evaluate the histogram of the dataframe
cancerDf.hist(bins = 10, figsize = (30,30), color = 'b')

# Prepare dataset for training
targetDf = cancerDf['Biopsy']
inputDf = cancerDf.drop(columns = ['Biopsy'])
X = np.array(inputDf).astype('float32')
y = np.array(targetDf).astype('float32')
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data in to test and train sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size = 0.5)

# Train XGBoost classifier model
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 500, n_estimators = 1000)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_predict, y_test)
plt.figure(figsize = (30,30))
sns.heatmap(cm, annot = True)
plt.show()