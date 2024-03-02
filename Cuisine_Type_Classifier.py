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

X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)

encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']]).toarray()

X = pd.concat([X.drop(['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein'], axis=1), pd.DataFrame(X_encoded)], axis=1)

X.columns = X.columns.astype(str)  # Remove leading/trailing whitespaces from column names

label_encoder_cuisine = LabelEncoder()
y_cuisine_type = label_encoder_cuisine.fit_transform(df['Cuisine_Type'])


X_train, X_test, y_train, y_test = train_test_split(X, y_cuisine_type, test_size=0.2, random_state=42)


model_cuisine_type = RandomForestClassifier(random_state=42)

model_cuisine_type.fit(X_train, y_train)

y_pred_cuisine = model_cuisine_type.predict(X_test)

# Evaluate the model
accuracy_cuisine = accuracy_score(y_test, y_pred_cuisine)
print(f'Model Accuracy (Cuisine Type): {accuracy_cuisine * 100:.2f}%')

# Output predictions for a few test cases
test_cases_cuisine = X_test.head(5)  # Adjust the number of test cases as needed
predicted_labels_cuisine = label_encoder_cuisine.inverse_transform(model_cuisine_type.predict(test_cases_cuisine))

'''
for i, (index, row) in enumerate(test_cases_cuisine.iterrows()):
    print(f'Test Case {i + 1}:')
    print(f'   Features: {row}')
    print(f'   Predicted Cuisine Type: {predicted_labels_cuisine[i]}')
    print()
'''