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
label_encoder_meal = LabelEncoder()
y_meal_type = label_encoder_meal.fit_transform(df['Meal_Type'])

# Split the data into training and testing sets for 'Meal_Type'
X_train_meal, X_test_meal, y_train_meal, y_test_meal = train_test_split(X, y_meal_type, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Meal_Type'
model_meal_type = RandomForestClassifier(random_state=42)
model_meal_type.fit(X_train_meal, y_train_meal)

# Predictions on the test set for 'Meal_Type'
y_pred_meal = model_meal_type.predict(X_test_meal)

# Evaluate the model for 'Meal_Type'
accuracy_meal = accuracy_score(y_test_meal, y_pred_meal)
print(f'Model Accuracy (Meal Type): {accuracy_meal * 100:.2f}%')

# Show predictions for a few test cases for 'Meal_Type'
test_cases_meal = X_test_meal.head()  # You can change this to any number of test cases you want to display
test_predictions_meal = label_encoder_meal.inverse_transform(y_pred_meal[:len(test_cases_meal)])
actual_labels_meal = label_encoder_meal.inverse_transform(y_test_meal[:len(test_cases_meal)])

print("\nMeal Type Predictions:")
for i, (actual, predicted) in enumerate(zip(actual_labels_meal, test_predictions_meal)):
    print(f"Test Case {i + 1}: Actual - {actual}, Predicted - {predicted}")

# Encode the target variable ('Cuisine_Type') using LabelEncoder
label_encoder_cuisine = LabelEncoder()
y_cuisine_type = label_encoder_cuisine.fit_transform(df['Cuisine_Type'])

# Split the data into training and testing sets for 'Cuisine_Type'
X_train_cuisine, X_test_cuisine, y_train_cuisine, y_test_cuisine = train_test_split(X, y_cuisine_type, test_size=0.2, random_state=42)

# Initialize Random Forest classifier for 'Cuisine_Type'
model_cuisine_type = RandomForestClassifier(random_state=42)
model_cuisine_type.fit(X_train_cuisine, y_train_cuisine)

# Predictions on the test set for 'Cuisine_Type'
y_pred_cuisine = model_cuisine_type.predict(X_test_cuisine)

# Evaluate the model for 'Cuisine_Type'
accuracy_cuisine = accuracy_score(y_test_cuisine, y_pred_cuisine)
print(f'Model Accuracy (Cuisine Type): {accuracy_cuisine * 100:.2f}%')

# Show predictions for a few test cases for 'Cuisine_Type'
test_cases_cuisine = X_test_cuisine.head()  # You can change this to any number of test cases you want to display
test_predictions_cuisine = label_encoder_cuisine.inverse_transform(y_pred_cuisine[:len(test_cases_cuisine)])
actual_labels_cuisine = label_encoder_cuisine.inverse_transform(y_test_cuisine[:len(test_cases_cuisine)])

print("\nCuisine Type Predictions:")
for i, (actual, predicted) in enumerate(zip(actual_labels_cuisine, test_predictions_cuisine)):
    print(f"Test Case {i + 1}: Actual - {actual}, Predicted - {predicted}")
