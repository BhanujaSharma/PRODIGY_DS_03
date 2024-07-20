import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")
# Update the file path to the correct location of your CSV file
file_path = r'C:\Users\Bhanuja\Desktop\BHANUJA\PROJECTS\bank-additional.csv'
# Load the dataset from the local file
df = pd.read_csv(file_path, delimiter=';')

# Rename the target column
df.rename(columns={'y': 'deposit'}, inplace=True)

# Check dataset information
print("Data Shape:", df.shape)
print("Columns:", df.columns)
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isna().sum())
print("Duplicate Values:", df.duplicated().sum())

# Handling Missing and Duplicate Values
df.drop_duplicates(inplace=True)

# Visualizing Numerical Columns
df.hist(figsize=(10, 10), color='red')
plt.show()

# Visualizing Categorical Data
cat_cols = df.select_dtypes(include='object').columns
for feature in cat_cols:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=feature, data=df, palette='crest')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

# Plotting Boxplot and Checking for Outliers
df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(20, 10), color='red')
plt.show()

# Removing Outliers
def remove_outliers(column):
    q1 = np.quantile(column, 0.25)
    q3 = np.quantile(column, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return column[(column >= lower_bound) & (column <= upper_bound)]

num_cols = ['age', 'campaign', 'duration']
for col in num_cols:
    df[col] = remove_outliers(df[col])

# Plotting Boxplot after Removing Outliers
df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(20, 10), color='red')
plt.show()

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df)

# Calculate and plot correlations
corr = df_encoded.corr()
high_corr = corr[abs(corr) >= 0.90]
sns.heatmap(high_corr, annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()

# Feature selection based on high correlation
high_corr_cols = ['emp.var.rate', 'euribor3m', 'nr.employed']
df1 = df_encoded.drop(columns=high_corr_cols)

# Convert categorical columns to numerical using LabelEncoder
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)

# Selecting Independent and Dependent Variables
x = df_encoded.drop('deposit', axis=1)
y = df_encoded['deposit']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Building and Training Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10)
dt.fit(x_train, y_train)

# Evaluating the Model
def eval_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy Score:", acc)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def mscore(model):
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("\nTraining Score:", train_score)
    print("\nTesting Score:", test_score)

# Generating Predictions
ypred_dt = dt.predict(x_test)

# Evaluate the Decision Tree Model
print("Evaluation for Decision Tree (criterion='gini', max_depth=5, min_samples_split=10)")
eval_model(y_test, ypred_dt)
mscore(dt)

# Generating Predictions
ypred_dt1 = dt1.predict(x_test)

# Evaluating the Second Decision Tree Model
print("Evaluation for Decision Tree (criterion='entropy', max_depth=4, min_samples_split=15)")
eval_model(y_test, ypred_dt1)
mscore(dt1)

# Plotting the Second Decision Tree
plt.figure(figsize=(20, 15))
plot_tree(dt1, feature_names=x.columns.tolist(), class_names=["no", "yes"], filled=True, fontsize=10)
plt.title('Decision Tree Visualization (Entropy)')
plt.show()
