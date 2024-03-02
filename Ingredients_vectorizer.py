from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('Task 01\\final_dataset.csv')
print(df.columns)
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespaces from column names

# Convert the 'Ingredients' column to a space-separated string
df['Ingredients'] = df['Ingredients'].apply(lambda x: [ingredient.strip(" '") for ingredient in x[1:-1].split(',')[:-1]])

# Convert the 'Ingredients' column to a space-separated string
df['Ingredients_str'] = df['Ingredients'].apply(lambda x: ' '.join(x))

print(df['Ingredients'])

# Use TfidfVectorizer to convert text data to numerical format
vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
X_ingredients = vectorizer.fit_transform(df['Ingredients_str'])

# Combine the ingredient features with other features
X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)

print(X)

#--------------------------------------------------------------------------------

#ENCODING OF DIET AND FLAVOUR

from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical columns 'Diet' and 'Flavor'
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Diet', 'Flavor']]))
X_encoded.columns = encoder.get_feature_names(['Diet', 'Flavor'])

# Concatenate the one-hot encoded columns with the rest of the features
X = pd.concat([X.drop(['Diet', 'Flavor'], axis=1), X_encoded], axis=1)

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











'''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import pandas as pd

# Assume df is the DataFrame containing the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')

df.columns = df.columns.str.strip() # Remove leading/trailing whitespaces from column names
print(df.columns)


# Preprocess 'Health_Benefits' column
df['Health_Benefits'] = df['Health_Benefits'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# Extract features and target variables
X = df[['Ingredients', 'Diet', 'Flavor', 'Protein', 'Carbohydrate_Content', 'Meal_Type', 'Cuisine_Type']]
y_health_benefits = MultiLabelBinarizer().fit_transform(df['Health_Benefits'])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=['Ingredients', 'Diet', 'Flavor', 'Meal_Type', 'Cuisine_Type'])

# Split the data into training and testing sets for each target variable
X_train, X_test, y_health_benefits_train, y_health_benefits_test = train_test_split(
    X_encoded, y_health_benefits, test_size=0.2, random_state=42
)

# Initialize Random Forest classifier for 'Health_Benefits'
model_health_benefits = RandomForestClassifier()

# Fit the model
model_health_benefits.fit(X_train, y_health_benefits_train)

# Predictions for a new dish
new_dish_attributes = extract_attributes(new_dish)
new_dish_features = pd.get_dummies(pd.DataFrame([new_dish_attributes], columns=X.columns),
                                   columns=['Ingredients', 'Diet', 'Flavor', 'Meal_Type', 'Cuisine_Type'])
new_dish_health_benefits = model_health_benefits.predict(new_dish_features)

# Convert predicted labels back to tuples
predicted_health_benefits = MultiLabelBinarizer().inverse_transform(new_dish_health_benefits)

# Combine predictions
final_categorization = {
    'Health_Benefits': predicted_health_benefits
}

print(final_categorization)
'''