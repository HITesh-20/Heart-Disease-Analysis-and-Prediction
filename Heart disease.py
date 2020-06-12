#HEART DISEASE PREDICTION
#IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#READ DATA
df=pd.read_csv("C:/Users/Hitesh/Desktop/MACHINE LEARNING/Logistic Regression/Heart Disease.csv")
print(df)

#DROPPING IRRELAVENT COLUMNS
df1=df.drop(['education','BPMeds','cigsPerDay','prevalentStroke','prevalentHyp'],axis='columns')
print(df.columns)

#DESCRIPTION
print("\n\nDESCRIPTION\n\n",df1.describe())

#CORRELATION
print(df1.corr())

#HANDLING MISSING VALUES
print("\n\nBEFORE HANDLING MISSING DATA\n",df1.isnull().sum())
df1.fillna(df.mean(),inplace=True)
print("\n\nAFTER HANDLING MISSING DATA\n",df1.isnull().sum())
print("\n\nSHAPE OF DATASET",df1.shape,"\n\n")

#SAVING FILE FOR EDA
df1.to_csv('Heart Disease1.csv')
print("SAVED SUCCESSFULLY")

#NOW I'LL DO EDA ON TABLEAU JUST JUMP TO TABBLEAU FILE
#AFTER THAT I'LL APPLY LOGISTIC REGRESSION BY SKLEARN HERE ONLY

print("\nLOGISTIC REGRESSION\n")
X=df1.drop('TenYearCHD',axis='columns')
y=df1.TenYearCHD

#SPLITTING TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("\n\nX_train\n\n",X_train)
print("\n\nX_test\n\n",X_test)
print("\n\ny_train\n\n",y_train)
print("\n\ny_test\n\n",y_test)

#FITTING MODEL
model=LogisticRegression().fit(X_train,y_train)
acc=model.score(X_test, y_test)
print("\nACCURACY :",acc*100)

#YOU CAN ALSO PREDICT FOR  ANY PERSON 
a=model.predict([[1,75,1,0,207,120,96,35.02,70,90.000021]])
print("PREDICTED RESULT OF RANDOM VALUES :",a)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
actual=y_test
predicted=model.predict(X_test)
print("\nCONFUSION MATRIX\n",confusion_matrix(actual,predicted))

#REPORT
from sklearn.metrics import classification_report
print("\nREPORT\n",classification_report(actual,predicted))