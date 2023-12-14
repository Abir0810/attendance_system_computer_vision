#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns 


# In[3]:


df=pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")


# In[4]:


df.info()


# In[5]:


df.head()


# # # Data Analysis

# In[6]:


from matplotlib import pyplot as plt


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
 
# Reading the tips.csv file
data = pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")
 
# initializing the data
x = data['t_c']
 
# plotting the data
plt.hist(x, bins, histtype='bar', rwidth=0.6,color='blue', align='right', edgecolor='black')
 
# Adding title to the plot
plt.title("Dataset")
 
# Adding label on the y-axis
plt.ylabel('Student number')
 
# Adding label on the x-axis
plt.xlabel('Total days')
 
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
 
# Reading the tips.csv file
data = pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")
 
# initializing the data
x = data['t_c']

bins = [5,10,15,20,25,30]
 
# plotting the data
plt.hist(x, bins=10, color='yellow', align='right', edgecolor='black')
 
# Adding title to the plot
plt.title("Dataset")
 
# Adding label on the y-axis
plt.ylabel('Student attendence')
 
# Adding label on the x-axis
plt.xlabel('Total days')
 
plt.show()


# In[2]:


import pandas as pd
data = pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")
import matplotlib.pyplot as xyz
x = data['t_c']
bins = [0,5,10,15,20,25,30]
xyz.hist(x, bins, histtype='bar', rwidth=0.9,color='yellow', align='right', edgecolor='black')

xyz.xlabel('Students attendence')
xyz.ylabel('Total days')
xyz.title('Dataset')
xyz.show()


# In[9]:


df.head()


# In[10]:


df = df[['s_name','t_c','enable']]
df.corr()


# In[11]:


plt.imshow(df.corr())


# In[12]:


plt.imshow(df.corr(), cmap="Spectral")
plt.colorbar()
plt.show()


# In[13]:


plt.barh(df.s_name, df.t_c)

plt.xlabel('Total attendence day')
plt.ylabel('Student name')
plt.title('Student Attendence days vizulaization')

plt.show()


# In[14]:


df=pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")


# In[15]:


df.head()


# In[16]:


plt.plot(df.c_8,df.s_name,color='Red',linewidth=5, linestyle='dotted')
plt.xticks(rotation=70, horizontalalignment="center")
plt.xlabel('Attendece')
plt.ylabel('')
plt.title('8th class attendence')


# In[17]:


plt.plot(df.s_name,df.c_24,color='blue',linewidth=5, linestyle='dotted')
plt.xticks(rotation=90, horizontalalignment="center",color="red")
plt.xlabel('Attendece')
plt.ylabel('')
plt.title('24th class attendence')


# In[18]:


plt.scatter(df.t_c,df['s_name'])
plt.xlabel('Attendece')
plt.ylabel('Student Name')
plt.title('Total Attendence')


# In[19]:


plt.scatter(df.c_10,df['s_name'])
plt.xlabel('')
plt.ylabel('Name')
plt.title('10th class attendence')


# # Algorithms

# In[20]:


df.head()


# In[21]:


from sklearn import preprocessing


# In[22]:


label_encoder = preprocessing.LabelEncoder()


# In[23]:


df['s_name']= label_encoder.fit_transform(df['s_name'])


# In[24]:


df['s_name'].unique()


# In[25]:


y = df[['enable']]
x = df.drop(['enable'],axis=1)
x=x.dropna()


# In[26]:


x.head(2)


# In[27]:


y.head(1)


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[ ]:





# # LogisticRegression

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


logmodel=LogisticRegression()


# In[32]:


logmodel.fit(x_train,y_train)


# In[33]:


predictions = logmodel.predict(x_test)


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


classification_report(y_test,predictions)


# In[36]:


from sklearn.metrics import confusion_matrix


# In[37]:


confusion_matrix(y_test,predictions)


# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


accuracy_score(y_test,predictions)


# In[40]:


logmodel.score(x_test,y_test)


# In[41]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[42]:


cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)


# In[43]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)


# In[44]:


disp.plot()
plt.show()


# In[45]:


from sklearn import metrics


# In[46]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[47]:


Precision = metrics.precision_score(y_test, predictions)


# In[48]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[49]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[50]:


F1_score = metrics.f1_score(y_test, predictions)


# In[51]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[52]:


y_pred_train = logmodel.predict(x_train)


# In[53]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[54]:


print('Training set score: {:.4f}'.format(logmodel.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(logmodel.score(x_test, y_test)))


# In[55]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[ ]:





# # Support Vector Machine 

# In[56]:


from sklearn import svm


# In[57]:


logmodel = svm.SVC()


# In[58]:


logmodel.fit(x_train, y_train)


# In[59]:


predictions = logmodel.predict(x_test)


# In[60]:


from sklearn.metrics import accuracy_score


# In[61]:


accuracy_score(y_test,predictions)


# In[62]:


from sklearn.metrics import classification_report


# In[63]:


classification_report(y_test,predictions)


# In[64]:


from sklearn.metrics import confusion_matrix


# In[65]:


confusion_matrix(y_test,predictions)


# In[66]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[67]:


cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)


# In[68]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)


# In[69]:


disp.plot()
plt.show()


# In[70]:


from sklearn import metrics


# In[71]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[72]:


Precision = metrics.precision_score(y_test, predictions)


# In[73]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[74]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[75]:


F1_score = metrics.f1_score(y_test, predictions)


# In[76]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[77]:


y_pred_train = logmodel.predict(x_train)


# In[78]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[79]:


print('Training set score: {:.4f}'.format(logmodel.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(logmodel.score(x_test, y_test)))


# In[80]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Decision Tree 

# In[81]:


from sklearn import tree


# In[82]:


logmodel = tree.DecisionTreeClassifier()


# In[83]:


logmodel.fit(x_train, y_train)


# In[84]:


predictions = logmodel.predict(x_test)


# In[85]:


from sklearn.metrics import accuracy_score


# In[86]:


accuracy_score(y_test,predictions)


# In[87]:


from sklearn.metrics import classification_report


# In[88]:


classification_report(y_test,predictions)


# In[89]:


from sklearn.metrics import confusion_matrix


# In[90]:


confusion_matrix(y_test,predictions)


# In[91]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[92]:


cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)


# In[93]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)


# In[94]:


disp.plot()
plt.show()


# In[95]:


from sklearn import metrics


# In[96]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[97]:


Precision = metrics.precision_score(y_test, predictions)


# In[98]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[99]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[100]:


F1_score = metrics.f1_score(y_test, predictions)


# In[101]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[102]:


y_pred_train = logmodel.predict(x_train)


# In[103]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[104]:


print('Training set score: {:.4f}'.format(logmodel.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(logmodel.score(x_test, y_test)))


# In[105]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # SGD Classifier

# In[106]:


from sklearn.linear_model import SGDClassifier


# In[107]:


clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)


# In[108]:


clf.fit(x_train, y_train)


# In[109]:


clf.coef_


# In[110]:


clf.intercept_


# In[111]:


predictions = clf.predict(x_test)


# In[112]:


from sklearn.metrics import accuracy_score


# In[113]:


accuracy_score(y_test,predictions)


# In[114]:


from sklearn.metrics import classification_report


# In[115]:


classification_report(y_test,predictions)


# In[116]:


from sklearn.metrics import confusion_matrix


# In[117]:


confusion_matrix(y_test,predictions)


# In[118]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[119]:


cm = confusion_matrix(y_test, predictions, labels=clf.classes_)


# In[120]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)


# In[121]:


disp.plot()
plt.show()


# In[122]:


from sklearn import metrics


# In[123]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[124]:


Precision = metrics.precision_score(y_test, predictions)


# In[125]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[126]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[127]:


F1_score = metrics.f1_score(y_test, predictions)


# In[128]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[129]:


y_pred_train = clf.predict(x_train)


# In[130]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[131]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))


# In[132]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Light GBM 

# In[133]:


import lightgbm as lgb


# In[134]:


cl = lgb.LGBMClassifier()


# In[135]:


cl.fit(x_train, y_train)


# In[136]:


predictions = cl.predict(x_test)


# In[137]:


from sklearn.metrics import classification_report


# In[138]:


classification_report(y_test,predictions)


# In[139]:


y_pred=cl.predict(x_test)


# In[140]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))


# In[141]:


from sklearn.metrics import accuracy_score


# In[142]:


accuracy_score(y_test,predictions)


# In[143]:


y_pred_train = clf.predict(x_train)


# In[144]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[145]:


print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))


# In[146]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


# In[147]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[148]:


cm = confusion_matrix(y_test, predictions, labels=cl.classes_)


# In[149]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=cl.classes_)


# In[150]:


disp.plot()
plt.show()


# In[151]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[152]:


from sklearn import metrics


# In[153]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[154]:


Precision = metrics.precision_score(y_test, predictions)


# In[155]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[156]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[157]:


F1_score = metrics.f1_score(y_test, predictions)


# In[158]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[159]:


y_pred_train = cl.predict(x_train)


# In[160]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[161]:


print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))


# In[162]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Neural Network

# In[163]:


from sklearn.naive_bayes import GaussianNB


# In[164]:


logmo =GaussianNB()


# In[165]:


logmo.fit(x_train, y_train)


# In[166]:


predictions = logmo.predict(x_test)


# In[167]:


from sklearn.metrics import accuracy_score


# In[168]:


accuracy_score(y_test,predictions)


# In[169]:


from sklearn.metrics import classification_report


# In[170]:


classification_report(y_test,predictions)


# In[171]:


from sklearn.metrics import confusion_matrix


# In[172]:


confusion_matrix(y_test,predictions)


# In[173]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[174]:


cm = confusion_matrix(y_test, predictions, labels=logmo.classes_)


# In[175]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmo.classes_)


# In[176]:


disp.plot()
plt.show()


# In[177]:


from sklearn import metrics


# In[178]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[179]:


Precision = metrics.precision_score(y_test, predictions)


# In[180]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[181]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[182]:


F1_score = metrics.f1_score(y_test, predictions)


# In[183]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# # Unsupervised learning

# In[184]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[185]:


df=pd.read_excel(r"E:\attendence\final attenden sheet.xlsx")


# In[186]:


df.head(2)


# In[187]:


xpos = np.arange(len(df.s_name))
xpos


# In[188]:


plt.bar(df.s_name,df.t_c,color='black',)
plt.xlabel('Student name')
plt.ylabel('Classes')
plt.title('Student attendence')
plt.tick_params(axis='x', rotation=80)


# In[189]:


plt.bar(df.s_name,df.enable,color='black',)
plt.xlabel('Student name')
plt.ylabel('Classes')
plt.title('Student attendence')
plt.tick_params(axis='x', rotation=80)


# In[190]:


from sklearn import preprocessing


# In[191]:


label_encoder = preprocessing.LabelEncoder() 


# In[192]:


df['s_name']= label_encoder.fit_transform(df['s_name']) 


# In[193]:


df['s_name'].unique()


# In[194]:


plt.scatter(df.s_name,df['t_c'])
plt.xlabel('Attendence')
plt.ylabel('Classes')
plt.title('Students')


# In[195]:


plt.scatter(df.s_name,df['enable'])
plt.xlabel('Attendence')
plt.ylabel('Classes')
plt.title('Students')


# In[196]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[197]:


km = AgglomerativeClustering(n_clusters=3)
y_predicted = km.fit_predict(df[['t_c']])
y_predicted


# In[198]:


df['cluster']=y_predicted
df.head(2)


# In[199]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.s_name,df1['t_c'],color='green')
plt.scatter(df2.s_name,df2['t_c'],color='red')
plt.scatter(df3.s_name,df3['t_c'],color='red')
plt.xlabel('Classes')
plt.ylabel('')
plt.title('Attendence situation')
plt.legend()


# In[200]:


km = AgglomerativeClustering(n_clusters=2)
y_predicted = km.fit_predict(df[['enable']])
y_predicted


# In[201]:


df['cluster']=y_predicted
df.head(2)


# In[202]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
plt.scatter(df1.enable,df1['s_name'],color='green')
plt.scatter(df2.enable,df2['s_name'],color='red')
plt.xlabel('Classes')
plt.ylabel('')
plt.title('Attendence situation')
plt.legend()


# In[203]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
plt.scatter(df1.s_name,df1['enable'],color='green')
plt.scatter(df2.s_name,df2['enable'],color='red')
plt.xlabel('Classes')
plt.ylabel('')
plt.title('Attendence situation')
plt.legend()


# In[ ]:




