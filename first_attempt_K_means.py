# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:43:45 2020
@author: Amrit Raj
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#basic plotting detailing...........

#set font size of labels on matplotlib plots
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)
#Plotting detailing ends here...........

dataset=pd.read_excel('Project_Data.xlsx')
#check=dataset.iloc[:, 13].values
x=dataset.iloc[:, 1:14].values
print(x)

#Getting rid of categorical data of region...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#x=pd.DataFrame(x) to view it as  a data frame
#Getting rid of categorical data of Gender...
category_Hot_Encoded=pd.get_dummies(x[:, 1])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 1, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#x=pd.DataFrame(x) to view it as  a data frame
#Now we remove the female categorical data to avoid the categorical data trap
x=np.delete(x, 15, 1)
#x=pd.DataFrame(x)...just to visualize#Getting rid of categorical data of Gender...
#Getting rid of categorical data of Goals...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
#Getting rid of categorical data of Seats...
category_Hot_Encoded=pd.get_dummies(x[:, 7])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 7, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded


#Separating the hobbies separate dy comma
x_copy=x#This is used to maintain a x copy in object form
x=pd.DataFrame(x)

#Hobbies
clist = ['Game','Musical','Gym','Others']
for i in range(0, 325):
    list1 = x[3][i].split(', ')
    list2 = [0, 0, 0, 0]
    
    for j in range(0, 4):
        if clist[j] in list1:
            list2[j] = 1 
    x[3][i] = list2
    
    
tags = x[3].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))

x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'Game'}, inplace = True)
x.rename(columns = {'tag_1':'Musical'}, inplace = True)
x.rename(columns = {'tag_2':'Gym'}, inplace = True)
x.rename(columns = {'tag_3':'Others'}, inplace = True)

#Delete column number 3(Hobbies) from the x...
x=x.drop(labels=3, axis=1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded
x=x.drop(labels=5, axis=1)# This removes the friends coulumm

#Music taste
clist = ['M1','M2','M3','M4','M5','M6','M7','M8','Others']
for i in range(0, 325):
    list1 = x[8][i].split(', ')
    list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(0, 9):
        if clist[j] in list1:
            list2[j] = 1
            
    x[8][i] = list2
    
    
tags = x[8].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))

x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)

x.rename(columns = {'tag_0':'M1'}, inplace = True)
x.rename(columns = {'tag_1':'M2'}, inplace = True)
x.rename(columns = {'tag_2':'M3'}, inplace = True)
x.rename(columns = {'tag_3':'M4'}, inplace = True)
x.rename(columns = {'tag_4':'M5'}, inplace = True)
x.rename(columns = {'tag_5':'M6'}, inplace = True)
x.rename(columns = {'tag_6':'M7'}, inplace = True)
x.rename(columns = {'tag_7':'M8'}, inplace = True)
x.rename(columns = {'tag_8':'Others'}, inplace = True)
x=x.drop(labels=8, axis=1)# This removes the Music coulumm

#This drop is only temporary and needs to be removed as and when nasha is implemented...
x=x.drop(labels=6, axis=1)# This removes the Nasha coulumm
x=x.drop(labels=2, axis=1)# This removes the Nasha coulumm
#.....



# Now, we will attempt to apply K-Means algorithm on x.......
from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
	kmeans=KMeans(n_clusters = i , init = 'k-means++', max_iter=200, n_init=10, random_state=0)
	kmeans.fit(x)
	wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow method to find the best value of k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#found--k=4.....
#We use k=4 and find the y_kmeans
kmeans=KMeans(n_clusters = 4 , init = 'k-means++', max_iter=200, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(x)

x=x.values

#Visualizing the results obtained by plotting scholar id vs income...
plt.scatter(x[y_kmeans==0, 0]%100, x[y_kmeans==0, 1], s=10, c='red')
plt.scatter(x[y_kmeans==1, 0]%100, x[y_kmeans==1, 1], s=10, c='blue')
plt.scatter(x[y_kmeans==2, 0]%100, x[y_kmeans==2, 1], s=10, c='green')
plt.scatter(x[y_kmeans==3, 0]%100, x[y_kmeans==3, 1], s=10, c='cyan')
plt.title('Group(Cluster) of friends based on K-Means')
plt.xlabel('Scholar_ID mod 100---->')
plt.ylabel('Spending Score----->')
plt.legend()
plt.show()		                        

#Counting the number of 0's to 3/s....
zero=0
one=0
two=0
three=0

for i in y_kmeans:
	if(i==1):
		one+=1
	elif(i==2):
		two+=1
	elif(i==3):
		three+=1
	elif(i==0):
		zero+=1
		
#Hence, these are the number of people in the respective groups

"""
#Visualization.......
plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1], s=75, c='red', label='cluster1')
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1], s=75, c='blue', label='cluster2')
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1], s=75, c='green', label='cluster3')
plt.scatter(x[y_kmeans==3, 0], x[y_kmeans==3, 1], s=75, c='cyan', label='cluster4')
plt.title('Group(Cluster) of friends based on K-Means')
plt.xlabel('Scholar_ID---->')
plt.ylabel('Spending Score----->')
plt.legend()
plt.show()
#Nasha
clist = ['A','G','C','X','N']
for i in range(0, 328):
    list1 = x[4][i].split(', ')
    list2 = [0, 0, 0, 0, 0]
    
    for j in range(0, 5):
        if clist[j] in list1:
            list2[j] = 1 
    x[6][i] = list2
    
    
tags = x[4].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))
x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'A'}, inplace = True)
x.rename(columns = {'tag_1':'G'}, inplace = True)
x.rename(columns = {'tag_2':'C'}, inplace = True)
x.rename(columns = {'tag_3':'X'}, inplace = True)
x.rename(columns = {'tag_4':'N'}, inplace = True)
"""

"""
#Making the data polynomial...
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg=PolynomialFeatures(degree=i)
	x_poly=poly_reg.fit_transform(x)
"""



"""
#for i in range(5):
	
	#Making the data polynomial...
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg=PolynomialFeatures(degree=4)
	x_poly=poly_reg.fit_transform(x)
	#Splitting the dataset...
	from sklearn.cross_validation import train_test_split
	x_train, x_test, y_train, y_test=train_test_split(x_poly, y, test_size=0.1, random_state=0)
	
	#applying feature scaling...
	from sklearn.preprocessing import StandardScaler
	sc=StandardScaler()
	x_train=sc.fit_transform(x_train)
	x_test=sc.transform(x_test)
	
	#fitting the linear regression model
	from sklearn.linear_model import LinearRegression
	regressor=LinearRegression()
	regressor.fit(x_train, y_train)
	#predicting the values
	y_pred=regressor.predict(x_test)
	
	loss=0
	err=0
	avg=0
	for j in range(77):
		print("Loss->",(y_test[j]-y_pred[j]))
		avg=avg+y_test[j]
		err=err+abs((y_test[j]-y_pred[j]))
		loss=loss+(y_test[j]-y_pred[j])*(y_test[j]-y_pred[j])
	cost=loss/77
	err=err/77
	avg=avg/77
	per_error=(err/avg)*100
	
	#plotting the result
	plt.scatter(check, y, color='red')
	plt.plot(check, regressor.predict(x), color='blue')
	plt.xlabel('AH--->')
	plt.ylabel('RH--->')
	plt.show()
	
"""
