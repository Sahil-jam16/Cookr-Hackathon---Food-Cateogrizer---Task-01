from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()

# Preprocess the 'Health_Benefits' column
df['Health_Benefits'] = df['Health_Benefits'].apply(lambda x: [benefit.strip(" '") for benefit in x[1:-1].split(',')])

# Use TfidfVectorizer to convert text data to numerical format
vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_benefits = vectorizer.fit_transform(df['Health_Benefits'].apply(lambda x: ' '.join(x)))

# Combine the benefits features with other features
X = pd.concat([df[['Diet', 'Flavor', 'Protein', 'Carbohydrate_Content']], pd.DataFrame(X_benefits.toarray())], axis=1)
X.columns = X.columns.astype(str)
# Define column transformer for one-hot encoding
categorical_cols = ['Diet', 'Flavor']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
    ],
    remainder='passthrough'
)

# Initialize MultiLabelBinarizer to one-hot encode the 'Health_Benefits' column
mlb = MultiLabelBinarizer()
y_health_benefits = mlb.fit_transform(df['Health_Benefits'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_health_benefits, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Health_Benefits'
model_health_benefits = RandomForestClassifier(random_state=42)

# Create a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model_health_benefits)
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy (Health Benefits): {accuracy * 100:.2f}%')

# Display some predictions
sample_indices = [0, 1, 2]  # You can modify these indices based on your dataset
for idx in sample_indices:
    row = X_test.iloc[idx]
    features = row.index
    values = row.values
    prediction = pipeline.predict([row])
    
    print(f'\nExample {idx + 1}:')
    print(f'   Features: {dict(zip(features, values))}')
    print(f'   Predicted Health Benefits: {mlb.inverse_transform(prediction)[0]}')


'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Diet', 'Flavor', 'Protein', 'Carbohydrate_Content'])

# Preprocess the 'Health_Benefits' column
df['Health_Benefits'] = df['Health_Benefits'].apply(lambda x: [benefit.strip(" '") for benefit in x[1:-1].split(',')])

# Convert the 'Health_Benefits' column to a space-separated string
df['Health_Benefits_str'] = df['Health_Benefits'].apply(lambda x: ' '.join(x))

# Use TfidfVectorizer to convert text data to numerical format
vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_benefits = vectorizer.fit_transform(df['Health_Benefits_str'])

# Combine the benefits features with other features
X = pd.concat([df.drop(['Health_Benefits'], axis=1), pd.DataFrame(X_benefits.toarray())], axis=1)

X.columns = X.columns.astype(str)
# Initialize MultiLabelBinarizer to one-hot encode the 'Health_Benefits' column
mlb = MultiLabelBinarizer()
y_health_benefits = mlb.fit_transform(df['Health_Benefits'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_health_benefits, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Health_Benefits'
model_health_benefits = RandomForestClassifier(random_state=42)

# Fit the model
model_health_benefits.fit(X_train, y_train)

# Predictions on the test set
y_pred = model_health_benefits.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy (Health Benefits): {accuracy * 100:.2f}%')
'''