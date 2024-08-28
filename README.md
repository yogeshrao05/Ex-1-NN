<H3>Name : YOGESH RAO S</H3>
<H3>REGISTER NO: 212222110055</H3>
<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### IMPORT LIBRARIES : 

```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### READ THE DATA: 
```py
df=pd.read_csv("Churn_Modelling.csv")
```

### CHECK DATA: 
```py
df.head()
df.tail()
df.columns
```

### CHECK THE MISSING DATA:
```py
df.isnull().sum()
```

### ASSIGNING X:
```py
X = df.iloc[:,:-1].values
X
```

### ASSIGNING Y:
```py
Y = df.iloc[:,-1].values
Y
```

### CHECK FOR OUTLIERS:
```py
df.describe()
```

### DROPPING STRING VALUES DATA FROM DATASET:
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

### CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```py
data.head()
```

### NORMALIE THE DATASET USING (MinMax Scaler):
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### SPLIT THE DATASET:
```py
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```

### TRAINING AND TESTING MODEL:
```py
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```



## OUTPUT:
### DATA CHECKING:
![image](https://github.com/user-attachments/assets/2f2af1a5-66bc-48b9-8298-99a5a104207b)



### MISSING DATA:
![image](https://github.com/user-attachments/assets/46347af4-3dae-45bd-b63e-c284fd5c0ff6)


### DUPLICATES IDENTIFICATION:
![image](https://github.com/user-attachments/assets/112b193c-5bb8-425a-92a7-8f30f5903949)




### VALUE OF Y:
![image](https://github.com/user-attachments/assets/62ea3cd7-0d7d-48e7-a153-4d4d9be6150c)


### OUTLIERS:
![image](https://github.com/user-attachments/assets/9dd99cc4-f6b5-432e-bc5b-f4b20b7aee22)


### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![image](https://github.com/user-attachments/assets/526f33a3-e989-4c82-8638-1e5ece4f0284)


### NORMALIZE THE DATASET:
![image](https://github.com/user-attachments/assets/3d4727b7-757c-4a57-b38e-1efaed72555e)


### SPLIT THE DATASET:
![image](https://github.com/user-attachments/assets/10e5ba79-f97f-44b7-8efa-8653048afc34)


### TRAINING AND TESTING MODEL:
![image](https://github.com/user-attachments/assets/a7c0cb51-6751-407f-aff1-bf409c7815f8)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


