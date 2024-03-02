from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# WORKING FOR SINGLE TARGET VARIABLE

'''

# Load the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespaces from column names

# Preprocess the 'Ingredients' column
df['Ingredients'] = df['Ingredients'].apply(lambda x: [ingredient.strip(" '") for ingredient in x[1:-1].split(',')[:-1]])

# Convert the 'Ingredients' column to a space-separated string
df['Ingredients_str'] = df['Ingredients'].apply(lambda x: ' '.join(x))

# Use TfidfVectorizer to convert text data to numerical format
vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_ingredients = vectorizer.fit_transform(df['Ingredients_str'])

# Combine the ingredient features with other features
X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)

# One-hot encode categorical columns 'Diet', 'Flavor', 'Carbohydrate_Content', and 'Protein'
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']]).toarray()

# Concatenate the one-hot encoded columns with the rest of the features
X = pd.concat([X.drop(['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein'], axis=1), pd.DataFrame(X_encoded)], axis=1)

X.columns = X.columns.astype(str)  # Remove leading/trailing whitespaces from column names

# Encode the target variable ('Meal_Type') using LabelEncoder
label_encoder = LabelEncoder()
y_meal_type = label_encoder.fit_transform(df['Meal_Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_meal_type, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Meal_Type'
model_meal_type = RandomForestClassifier(random_state=42)

# Fit the model
model_meal_type.fit(X_train, y_train)

# Predictions on the test set
y_pred = model_meal_type.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy (Meal Type): {accuracy * 100:.2f}%')

'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespaces from column names

# Preprocess the 'Ingredients' column
df['Ingredients'] = df['Ingredients'].apply(lambda x: [ingredient.strip(" '") for ingredient in x[1:-1].split(',')[:-1]])

# Convert the 'Ingredients' column to a space-separated string
df['Ingredients_str'] = df['Ingredients'].apply(lambda x: ' '.join(x))

# Use TfidfVectorizer to convert text data to numerical format
vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_ingredients = vectorizer.fit_transform(df['Ingredients_str'])

# Combine the ingredient features with other features
X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)

# One-hot encode categorical columns 'Diet', 'Flavor', 'Carbohydrate_Content', and 'Protein'
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']]).toarray()

# Concatenate the one-hot encoded columns with the rest of the features
X = pd.concat([X.drop(['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein'], axis=1), pd.DataFrame(X_encoded)], axis=1)

X.columns = X.columns.astype(str)  # Remove leading/trailing whitespaces from column names

# Encode the target variable ('Meal_Type') using LabelEncoder
label_encoder = LabelEncoder()
y_meal_type = label_encoder.fit_transform(df['Meal_Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_meal_type, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Meal_Type'
model_meal_type = RandomForestClassifier(random_state=42)

# Fit the model
model_meal_type.fit(X_train, y_train)

# Predictions on the test set
y_pred = model_meal_type.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy (Meal Type): {accuracy * 100:.2f}%')

# Show predictions for a few test cases
test_cases = X_test.head()  # You can change this to any number of test cases you want to display
test_predictions = label_encoder.inverse_transform(y_pred[:len(test_cases)])
actual_labels = label_encoder.inverse_transform(y_test[:len(test_cases)])

print("\nTest Cases:")
for i, (actual, predicted) in enumerate(zip(actual_labels, test_predictions)):
    print(f"Test Case {i + 1}: Actual - {actual}, Predicted - {predicted}")
