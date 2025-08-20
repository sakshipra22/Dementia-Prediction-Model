import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="whitegrid")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Models
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Tunning RFC with cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix #Confusion matrix

df = pd.read_excel('oasis_longitudinal_demographics.xlsx')
print(df.head())

#Data Cleaning 
# print(df['Age'].isnull().sum())
print(df.describe(include=[np.number])) # lets see the summary stats of numerical columns
print(df.describe(include=[np.object])) # Catergorical column
df = df.drop(['Subject ID','MRI ID','Hand'],axis=1) #dropping irrelevant column
print(df.head())

print(df.isna().sum()) # checking missing values in each column
# for better understanding lets check the percentage of missing values in each column
round(df.isnull().sum()/len(df.index), 2)*100

#Plotting distribution of SES
def univariate_mul(var):
    fig = plt.figure(figsize=(16,12))
    cmap=plt.cm.Blues
    cmap1=plt.cm.coolwarm_r
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(212)
    ax2 = sns.distplot(df[[var]], hist=False)
    ax2 = sns.distplot(df[var], hist=False)
    df[var].plot(kind='hist',ax=ax1, grid=True)
    ax1.set_title('Histogram of '+var, fontsize=14)
    
    ax2=sns.distplot(df[[var]],hist=False)
    ax2.set_title('Distribution of '+ var)
    plt.show()

# univariate_mul('SES')   # lets see the distribution of SES to decide which value we can impute in place of missing values.
# print(df['SES'].describe())
# df['SES'].fillna((df['SES'].median()), inplace=True) #imputing missing value in SES with median
df.SES.fillna ( df.SES.mode() [0], inplace=True ) # impute mode
df.MMSE.fillna ( df.MMSE.mean() , inplace=True ) # impute mean
df.isna().sum()
#Analysing MMSE
# univariate_mul('MMSE')
def univariate_plot(var1, var2):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Plot histogram for SES
    sns.histplot(df[var1], kde=False, ax=axes[0], color='skyblue', label=var1)
    axes[0].set_title('Histogram of ' + var1, fontsize=14)
    axes[0].grid(True)
    axes[0].legend()

    # Plot distribution for SES
    sns.kdeplot(df[var1], fill=True, ax=axes[1], color='orange', label=var1)
    axes[1].set_title('Distribution of ' + var1, fontsize=14)
    axes[1].legend()

    # Plot histogram for MMSE on the same axes
    sns.histplot(df[var2], kde=False, ax=axes[0], color='lightcoral', label=var2)
    axes[0].legend()

    # Plot distribution for MMSE on the same axes
    sns.kdeplot(df[var2], fill=True, ax=axes[1], color='green', label=var2)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Display SES and MMSE distributions together
univariate_plot('SES', 'MMSE')
print(df['MMSE'].describe())
df['MMSE'].fillna((df['MMSE'].median()), inplace=True) #imputing missing value in SES with median
print(round(df.isnull().sum()/len(df.index), 2)*100)


# Defining function to create pie chart and bar plot as subplots
# Assuming 'Group' is a column in your DataFrame df
def plot_piechart(var):
    plt.figure(figsize=(14, 7))
    
    # Plotting Pie Chart
    plt.subplot(121)
    label_list = df[var].unique().tolist()
    df[var].value_counts().plot.pie(autopct="%1.0f%%", colors=sns.color_palette("prism", 7),
                                    startangle=60, labels=label_list,
                                    wedgeprops={"linewidth": 2, "edgecolor": "k"}, shadow=True)
    plt.title("Distribution of " + var + " variable")

    # Plotting Bar Chart
    plt.subplot(122)
    ax = df[var].value_counts().plot(kind="barh")

    for i, j in enumerate(df[var].value_counts().values):
        ax.text(.7, i, j, weight="bold", fontsize=12)

    plt.title("Count of " + var + " cases")
    plt.show()

# Calling the function with 'Group' as an argument
plot_piechart('Group')
print(df['CDR'].describe()) #Analyzing CDR

# Plotting CDR with other variable
def univariate_percent_plot(cat):
    fig = plt.figure(figsize=(18,12))
    cmap=plt.cm.Blues
    cmap1=plt.cm.coolwarm_r
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    
    result = df.groupby(cat).apply (lambda group: (group.CDR == 'Normal').sum() / float(group.CDR.count())
         ).to_frame('Normal')
    result['Dementia'] = 1 -result.Normal
    result.plot(kind='bar', stacked = True,colormap=cmap1, ax=ax1, grid=True)
    ax1.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)
    ax1.set_ylabel('% Dementia status (Normal vs Dementia)')
    ax1.legend(loc="lower right")
    group_by_stat = df.groupby([cat, 'CDR']).size()
    group_by_stat.unstack().plot(kind='bar', stacked=True,ax=ax2,grid=True)
    ax2.set_title('stacked Bar Plot of '+ cat +' (in %)', fontsize=14)
    ax2.set_ylabel('Number of Cases')
    plt.show()
def cat_CDR(n): # Categorizing feature
    if n == 0:
        return 'Normal'
    
    else:        # As we have no cases of sever dementia CDR score=3
        return 'Dementia'

df['CDR'] = df['CDR'].apply(lambda x: cat_CDR(x))
plot_piechart('CDR')

# Categorizing feature MMSE
def cat_MMSE(n):
    if n >= 24:
        return 'Normal'
    elif n <= 9:
        return 'Severe'
    elif n >= 10 and n <= 18:
        return 'Moderate'
    elif n >= 19 and n <= 23:   # As we have no cases of sever dementia CDR score=3
        return 'Mild'

df['MMSE'] = df['MMSE'].apply(lambda x: cat_MMSE(x))
plot_piechart('MMSE')

univariate_percent_plot('MMSE')

univariate_mul('Age')
print(df['Age'].describe())
df['age_group'] = pd.cut(df['Age'], [50 ,60, 70, 80, 90, 100], labels=['50-60', '60-70', '70-80', '80-90', '90-100'])
df['age_group'].value_counts()
print(df['age_group'].isnull().sum())
rows_with_missing_age_group = df[df['age_group'].isnull()]
print(rows_with_missing_age_group)
univariate_percent_plot('age_group') # Now plotting age group to see dementia distribution

# Bivariate Analysis
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="Age",hue="CDR",split=True, data=df)
plt.show()
print(df['eTIV'].describe())
# Estimated total intracranial volume (eTIV):
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="age_group", y="eTIV",hue="CDR",split=True, data=df)
plt.show()

plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="eTIV",hue="CDR",split=True, data=df)
plt.show()

plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="nWBV",hue="CDR",split=True, data=df)
plt.show()

print(df['EDUC'].describe())

plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="EDUC",hue="CDR",split=True, data=df)
plt.show()
print(df['SES'].describe())

# Now plotting socio economic status to see dementia distribution
univariate_percent_plot('SES')

plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="SES",hue="CDR",split=True, data=df)
plt.show()
# It suggests that womens have less dementia probability at extreme higher and extreme lower level of socio economic status while mens have exactly opposite phenomenon
print(df['ASF'].describe()) # ASF - Atlas scaling factor (unitless).
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="M/F", y="ASF",hue="CDR",split=True, data=df)
plt.show()

plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="MMSE", y="ASF",split=True, data=df)
plt.show()
# From the above plot we can get the intuition about ASF as in case of normal patients the value of ASF distributed between 0.8 and 1.6 but as the patients started showing dementia cases its value centered around 1 as in case of Mild, Moderate and Severe it shrinks down to 1.1
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x="MMSE", y="nWBV",split=True, data=df)
plt.show()
# Same pattern observed in case of nWBV as the dementia level increases nWBV centered between 0.65 and 0.70.
plt.figure(figsize=(12, 8)) 
ax = sns.violinplot(x="MMSE", y="Visit",split=True, data=df)
plt.show()
#Observation: Severe Dementia cases starts reporting as the number of visits increases to more than 3 whereas normal cases are also reported after higher number of visits more than 3 but they are very few in number.

# Multicollinearity
numeric_columns=df.select_dtypes(include=['float64','int64'])
plt.figure(figsize=(14, 8)) # As we can see Visit and MR Delay are showing close correlation to 0.92 but I am not dropping any correlated variable as of now.
sns.heatmap(numeric_columns.corr(), annot=True)
plt.show()

# Observation
# Most of the cases of dementia observed in the age group of 70 - 80 years of Age.
# Mens develop dementia at early age before 60 years while womens have tendency of dementia at later age of later than 60 years
# In mens dementia starts at an education level of 4 years and most prevalent at education level of 12 years and 16 years and it can also extend upto more than 20 years of education level, while in womens dementia starts after 5 years of education level and most prevalent around 12 to 13 years of education level and it started to decrease as womens education level increase
# Dementia is prevalent in Mens having highest and lowest socio economic status while womens having medium socio economic status have higher dementia cases.
# Lower values of ASF close to 1 corresponds to severe dementia cases.
# Severe dementia is diagnosed after minnimum 3 number of visits.
# profile = ProfileReport(df, config_file=False)  # Use the ProfileReport class
# profile.to_widgets()
# profile.to_file("oasis_longitudinal_demographics.html")
print("Unique values in MMSE column:", df['MMSE'].unique())
print("Unique values in CDR column:", df['CDR'].unique())
mmse_mapping = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
cdr_mapping = {'Normal': 0, 'Dementia': 1}

# Apply mapping to 'MMSE' and 'CDR' columns
df['MMSE'] = df['MMSE'].map(mmse_mapping).astype(float)
df['CDR'] = df['CDR'].map(cdr_mapping).astype(float)
#Pre Processing

df.rename ( columns = { 'Group': 'Dementia'}, inplace=True ) # rename
df.rename ( columns = { 'M/F': 'Sex'}, inplace=True ) # rename
# df['MMSE'] = pd.to_numeric(df['MMSE'], errors='coerce').astype(float)
# df['CDR'] = pd.to_numeric(df['CDR'], errors='coerce').astype(float)
# Store original non-null values
mmse_original_non_null = df['MMSE'].notnull()
cdr_original_non_null = df['CDR'].notnull()

# Replace non-numeric values in 'MMSE' and 'CDR' columns with NaN
df['MMSE'] = pd.to_numeric(df['MMSE'], errors='coerce')
df['CDR'] = pd.to_numeric(df['CDR'], errors='coerce')

# Restore original non-null values
df.loc[mmse_original_non_null, 'MMSE'] = df.loc[mmse_original_non_null, 'MMSE'].astype(float)
df.loc[cdr_original_non_null, 'CDR'] = df.loc[cdr_original_non_null, 'CDR'].astype(float)


# Management of categorical data
le = LabelEncoder ()
df.Sex = le.fit_transform ( df.Sex.values )
print ( 'Sex:\n0 : %s \n1 : %s\n\n' %(le.classes_[0], le.classes_[1]) )
df.Dementia = le.fit_transform ( df.Dementia.values )
print ( 'Dementia:\n0 : %s \n1 : %s \n2 : %s' %(le.classes_[0], le.classes_[1], le.classes_[2]) )

# Management of categorical data
df['Sex'] = df['Sex'].astype('category')
df['Dementia'] = df['Dementia'].astype('category')
# Separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['category']).columns

# Impute missing values in numeric columns with the mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# You can choose an appropriate strategy for categorical columns, e.g., fill with the most frequent value
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# remove age_group column
df=df.drop('age_group',axis=1)
# Convert 'eTIV' to integer type
df['eTIV'] = df['eTIV'].astype(np.int64)

df.info()
# Split in train and test set
X, y = df.drop ('Dementia', axis=1).values , df.Dementia.values
X_train, X_test, y_train, y_test = train_test_split ( X, y,
                                                     test_size = 0.2,
                                                     random_state = 1,
                                                     stratify = y)


# Resampling Train set
print ('Number of observations in the target variable before oversampling of the minority class:', np.bincount (y_train) )

smt = SMOTE ()
X_train, y_train = smt.fit_resample (X_train, y_train)

print ('\nNumber of observations in the target variable after oversampling of the minority class:', np.bincount (y_train) )

# Standardization of features
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform ( X_train )
X_test_std = std_scaler.transform ( X_test )

# Model
clf = [ LogisticRegression(random_state=42), DecisionTreeClassifier(random_state=42), SVC (random_state=42),
       RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42) ]
models = [ 'Logistic Regression', 'Tree', 'Support vector machine', 'RFC', 'Gradient boost' ]

for clf, model in zip(clf,models):
  clf.fit ( X_train_std, y_train )
  y_pred = clf.predict ( X_test_std )
  print ( f'Cross validation score of {model}: %.3f \n' %cross_val_score (clf, X_train_std, y_train, cv=5).mean() )

#   Tuning RFC with cross-validation
rfc = RandomForestClassifier(n_jobs=-1, random_state=42) 

param_grid = { 
    'n_estimators': [500, 700, 900],
    'min_samples_split': [2,4,6,8,10]
}

gs = GridSearchCV ( estimator = rfc,
                   param_grid = param_grid,
                   scoring = 'accuracy',
                   cv = 5,
                   refit = True,
                   n_jobs = -1
                   )

gs = gs.fit ( X_train_std, y_train )

print ( 'Parameter setting that gave the best results on the hold out data:', gs.best_params_ )
print ( 'Mean cross-validated score of the best_estimator: %.3f' %gs.best_score_ )

gs = gs.best_estimator_

gs.fit ( X_train_std, y_train )
y_pred = gs.predict ( X_test_std )
print ( f'Accuracy train score: %.4f' %gs.score (X_train_std, y_train) )
print ( f'Accuracy test score: %.4f' %accuracy_score ( y_test, y_pred ) )
#Confusion matrix
conf_matrix = confusion_matrix (  y_test, y_pred )

print ('Number of records in the test dataset: %d\n' %y_test.shape[0])

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
#plot 1
sns.heatmap(conf_matrix,ax=axes[0],annot=True, cmap='Blues', cbar=False, fmt='d')
axes[0].set_xlabel('\nPredicted label', size = 14)
axes[0].set_ylabel('True label\n', size = 14)

# plot 2
sns.heatmap(conf_matrix/np.sum(conf_matrix),ax=axes[1], annot=True, 
            fmt='.2%', cmap='Blues', cbar=False)
axes[1].set_xlabel('\nPredicted label', size = 14)
axes[1].set_ylabel('True label\n', size = 14)
axes[1].yaxis.tick_left()
plt.show()
# Result:
# The test dataset contains 75 records. From the confusion matrix it is concluded that:

# All subjects who do not suffer from dementia are correctly classified.

# All bellies suffering from dementia are also correctly classified.

# The model makes more mistakes only in classifying those borderline subjects, that is, those subjects who were not initially classified as demented but who became so during the data collection. In particular, 2.67% of them are labeled as having dementia from the beginning and 4% are classified as having no dementia from the beginning of the survey to the end.