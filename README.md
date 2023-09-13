# Mini-Project--Application-of-NN

<h1 align="center">Prediction of Liver Cirrhosis</h1>

## Project Description:
### **About Liver Cirrhosis**
* Chronic liver damage from a variety of causes leading to scarring and liver failure.
* Hepatitis and chronic alcohol abuse are frequent causes.
* Liver damage caused by cirrhosis can't be undone, but further damage can be limited.
* Initially patients may experience fatigue, weakness and weight loss.
* During later stages, patients may develop jaundice (yellowing of the skin), gastrointestinal bleeding, abdominal swelling and confusion.
### **About the dataset**
* This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. 
* The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). 
* This data set contains 441 male patient records and 142 female patient records.
### **Goal of the Project**
* The main aim of the project is to create a ANN model which classifies patients as Infected or not infected based on various protiens in the blood.
* By using the simple blood tests we can predict whether he is infected or not.
## Algorithm:
1. Import the Libraries.
2. Read the Dataset.
3. Check for Null Values, if there are any fill them.
4. Check for duplicated values, if there are any remove them.
5. Transform Categorical into Numerical values.
6. Check Correlation Values for each feature.
7. Drop UnCorrelated Featuers.
8. Assign X and Y.
9. Split Dataset into testing and training.
10. Apply MLP Classifier and predict accuracy
11. Analyze the metrics.
12. Predict for a given input
## Program:
<h3 align="center">Import the Libraries</h3>

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
<h3 align="center">Read & Basic info about Dataset</h3>

```py
df = pd.read_csv("./Liver.csv")
df
df.info()
df.describe()
df.columns
```
<h3 align="center">Check for Null Values & Remove them</h3>

```py
df.isnull().sum()
df['Albumin_and_Globulin_Ratio'] = 
   df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())
df.isnull().sum()
```
<h3 align="center">Check for Duplicate Values & Remove them</h3>

```py
print("Duplicate Values =",df.duplicated().sum())
df[df.duplicated()]
df=df.drop_duplicates()
df.duplicated().sum()
```
<h3 align="center">Encode Values</h3>

```py
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df['Dataset']=df['Dataset'].map({1:1,2:0})
df
```
<h3 align="center">Correlation Values</h3>

```py
plt.figure(figsize=(10,5))
df.corr()['Dataset'].sort_values(ascending=False).plot(kind='bar',color='black')
plt.xticks(rotation=90)
plt.xlabel('Variables in the Data')
plt.ylabel('Correlation Values')
plt.show()

df = df.drop(["Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"],axis=1)
df
```
<h3 align="center">Assigning X and Y</h3>

```py
X = df.drop(['Dataset'], axis=1)
X
y = df['Dataset']
y
```
<h3 align="center">Splitting Dataset</h3>

```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
print("Training sample shape =",X_train.shape)
print("Testing sample sample =",X_test.shape)
```

<h3 align="center">Creating MLP</h3>

```py
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier

reg = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.0001, max_iter=10000)  
reg.fit(X_train, y_train)

log_predicted= reg.predict(X_test)
```

<h3 align="center">Testing Metrics</h3>

```py
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")
print('Classification Report: \n', classification_report(y_test,log_predicted))
```

<h3 align="center">Testing Custom Inputs</h3>

```py
pred_0 = reg.predict([[25,0,0.1,0.1,44,4,8]])
pred_1 = reg.predict([[50,1,5,1,200,50,50]])
if(pred_0 == 1 or pred_1 ==1):
  print("Infected with Liver Cirrohisis")
else:
  print("Not Infected with Liver Cirrohisis")
```

## Output:

<h3 align="center">Read & Basic info about Dataset</h3>

#### Dataset
![image](https://user-images.githubusercontent.com/93427237/205485806-e91a292d-7378-4246-92b4-a45094723f52.png)

#### Info
<img width = 375 src="https://user-images.githubusercontent.com/93427237/205485793-1ecdb537-e2e4-440a-9e93-d4d1ee675255.png"></img>

</br>
</br>

#### Descrption
![image](https://user-images.githubusercontent.com/93427237/205485815-7e905927-3ff4-46ee-8c02-3d7e65797ab0.png)

#### Columns
![image](https://user-images.githubusercontent.com/93427237/205485791-8d00d4b7-215b-42d3-ac75-868f46636b5b.png)

<h3 align="center">Check for Null Values & Remove them</h3>

#### Null Value - Before Removing
![image](https://user-images.githubusercontent.com/93427237/205485776-62fdbb80-bad1-4f99-b263-33bfab29dfcb.png)
#### Null Value - After Removing
![image](https://user-images.githubusercontent.com/93427237/205485784-c8f2ba98-de0e-42be-a80e-7211011575d3.png)

<h3 align="center">Check for Duplicate Values & Remove them</h3>

#### Total Duplicate Values
![image](https://user-images.githubusercontent.com/93427237/205485745-c0dd9bcd-07fd-46b2-aef7-4ab9b7298f1f.png)
![image](https://user-images.githubusercontent.com/93427237/205485765-59d4ab37-f519-4741-b22f-8bc8b13093b2.png)

#### Duplicate Values - After Removing
![image](https://user-images.githubusercontent.com/93427237/205485739-c946f2bf-0e16-4350-9524-149699e4efb4.png)

<h3 align="center">Encode Values</h3>

#### Afer Encoding
![image](https://user-images.githubusercontent.com/93427237/205483239-61d83fe4-a414-457d-94e9-4f29d04b5064.png)

</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>

<h3 align="center">Correlation Values</h3>

#### Correlation

![image](https://user-images.githubusercontent.com/93427237/205483210-f9c4ea6e-1cf0-41c3-81ff-6300ec9212a8.png)

#### Dataset after dropping uncorrelated values
![image](https://user-images.githubusercontent.com/93427237/205483223-81baa694-fa42-478c-8fad-610b72c36d57.png)

</br>
</br>
</br>
</br>

<h3 align="center">Splitting Dataset</h3>

#### Training and testing size
![image](https://user-images.githubusercontent.com/93427237/205483194-c72c5bef-7e02-403e-be4a-e908459b8988.png)

<h3 align="center">Testing Metrics</h3>

#### Accuracy
![image](https://user-images.githubusercontent.com/93427237/205483171-3739dd2e-f1cb-41c4-b2bc-fac0e227d976.png)

#### Confusion Matrix
![image](https://user-images.githubusercontent.com/93427237/205483106-532a9ecd-6b45-4ccf-a8c6-0c68bc2c1017.png)

![image](https://user-images.githubusercontent.com/93427237/205483100-236647f7-a02d-4d81-8cfa-fc42c4cae829.png)

#### Classification Report
![image](https://user-images.githubusercontent.com/93427237/205483091-af6cca06-b7d8-4dca-a46a-e1c0e5bf6bd1.png)


<h3 align="center">Testing Custom Inputs</h3>

**Normal Levels** 
* **Total bilirubin**: 0.1 to 1.2 mg/dL
* **Direct bilirubin**: less than 0.3 mg/dL
* **Alkaline_Phosphotase** -44 to 147 international units per liter
* **Alamine_Aminotransferase** - 4 to 36 U/L
* **Aspartate_Aminotransferase** - 8 to 33 U/L.
#### Test -1
* Age = 25
* Gender = 0
* Total_Bilirubin = 0.1
* Direct_Bilirubin = 0.1
* Alkaline_Phosphotase = 44
* Alamine_Aminotransferase = 4
* Aspartate_Aminotransferase = 8

![image](https://user-images.githubusercontent.com/93427237/205483076-d74d41a8-2cf2-48dc-a2a0-c9fdbf4ca257.png)

#### Test- 2
* Age = 50
* Gender = 1
* Total_Bilirubin = 5
* Direct_Bilirubin = 1
* Alkaline_Phosphotase = 200
* Alamine_Aminotransferase = 50
* Aspartate_Aminotransferase = 50

![image](https://user-images.githubusercontent.com/93427237/205483065-281e62a1-ebfa-42ea-8040-d5221c247a0b.png)
## Advantage :
* This model is very helpful in predicting Liver Cirrohsis with a Blood Test only.
* Usually it invloves MRI or Scan to make sure.
* Thus it makes the test cost effective and more guaranteed.
* **75%** is a good accuracy score and it can further be increased by using certain Hyperparameters and Regularizing the ANN.
* These measures can be implemented in the next steps and our model will be more accuracte.
## Result:
Thus a MLP is trained to classify whether a patient is infected with Liver Cirrohsis or Not based various blood test results with nearly **75%(74.269%)** accuracy 
Refer Colab File <a href = "https://colab.research.google.com/drive/1yTXT1njguDQiC7B_c83ppTQaFFpxPNve#scrollTo=9-iVZD6qV9jx">HERE</a>
<h2 align="right">A Project By:</h1>
<h3 align="right">Shafeeq Ahamed.S - 212221230092</h3>
<h3 align="right">Sanjay Kumar.S.S - 212221240048</h3>
