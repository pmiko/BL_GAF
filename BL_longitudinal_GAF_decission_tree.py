
import pandas as pd
from sklearn.model_selection import  KFold,  GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.tree import export_graphviz
import graphviz

# make df

df_clin = pd.read_csv('', sep= ';', decimal=',')

# Select all columns of type 'object'
object_columns = df_clin.select_dtypes(include='object')

# Drop specified columns from the selection
object_columns = object_columns.drop(['mnppsd', 'Site', 'Functional_Impairment_60'], axis=1, errors='ignore')

# Replace ' ' with np.nan in these columns
df_clin[object_columns.columns] = object_columns.replace(' ', np.nan)

# Replace NaN with mode for each column
for column in df_clin.columns:
    mode_value = df_clin[column].mode()[0]  # Calculate the mode of the column
    df_clin[column] = df_clin[column].fillna(mode_value) 
    
# Convert the object columns to integers
df_clin[object_columns.columns] = df_clin[object_columns.columns].astype(int)
    
df_clin['Site'] = df_clin['Site'].astype(str)
non_numeric_columns = df_clin.select_dtypes(include=['object']).columns
non_numeric_columns = non_numeric_columns.drop(['mnppsd', 'Site', 'Functional_Impairment_60'], errors='ignore')


pd.set_option('display.max_rows', None)

label_encoders = {}
for column in non_numeric_columns:
    le = LabelEncoder()
    df_clin[column] = le.fit_transform(df_clin[column])
    label_encoders[column] = le  # Store encoders if needed for later
    
non_numeric_columns = df_clin.select_dtypes(include=['object']).columns    
print("Non-numeric columns:", non_numeric_columns)
 
print(df_clin.dtypes)


#Rename columns

df_clin = df_clin.rename(columns={'IDSC_5': 'IDSC_5: Mood (sad)'})

df_clin = df_clin.rename(columns={'FAST_22': 'FAST22: Being able to defend your interests'})
df_clin = df_clin.rename(columns={'FAST_19': 'FAST19: Having  good relationships with people close you'})
df_clin = df_clin.rename(columns={'FAST_12': 'FAST12: Ability to solve a problem adequately'})
df_clin = df_clin.rename(columns={'FAST_6': 'FAST6: Accomplishing tasks as quickly as necessary'})
df_clin = df_clin.rename(columns={'FAST_5': 'FAST5: Holding down a paid job'})
df_clin = df_clin.rename(columns={'FAST_7': 'FAST7: Working in the field in which you were educated'})
df_clin = df_clin.rename(columns={'FAST_17': 'FAST17: Maintaining a friendship or friendships'})
df_clin = df_clin.rename(columns={'GAF_present': 'GAF_baseline'})


features_SVM = [
    'FAST22: Being able to defend your interests',
    'FAST19: Having  good relationships with people close you',
    'FAST12: Ability to solve a problem adequately',
    'FAST6: Accomplishing tasks as quickly as necessary',
    'FAST5: Holding down a paid job',
    'FAST7: Working in the field in which you were educated',
    'FAST17: Maintaining a friendship or friendships',
    'IDSC_5: Mood (sad)',
    'Anxiety_Disorder',
    'PQ16',
    'GAF_baseline',
    'GAF_maximum'
    ]


df_clin['Functional_Impairment_60'] = df_clin['Functional_Impairment_60'].replace({
    True: 'Impaired',
    False: 'Non-impaired' })


X = df_clin[features_SVM] #features_SVM , df_clin.drop(columns=['Functional_Impairment_60', 'mnppsd', 'Site', 'GAF_Deterioration_60','GAF_FU2_60', 'GAF_FU4_60' ])
y = df_clin['Functional_Impairment_60']

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grid for cross-validation
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [3],
    'min_samples_split': [2, 5, 10], 
    'class_weight': ['balanced']
}

# Perform 5-fold cross-validation
clf = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),  # Explicitly specify `estimator`
    param_grid=param_grid,
    cv=kf,
    scoring='balanced_accuracy',
    n_jobs=-1
)
clf.fit(X, y) 

# Store the best model
clf = clf.best_estimator_

print("Best parameters:", clf.get_params())

# Evaluate model performance using cross-validation
y_pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
balanced_acc = balanced_accuracy_score(y, y_pred)
print("Balanced Accuracy:", balanced_acc)

# Export the decision tree in DOT format
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=[str(cls) for cls in clf.classes_],  # Convert classes to string
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True
)

# Use graphviz to render the tree
graph = graphviz.Source(dot_data)
graph.render("", directory='C:/',  format="pdf", cleanup=True)

# Display the decision tree
graph.view()


