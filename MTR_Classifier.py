from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv('Task 01\\final_dataset.csv')
df.columns = df.columns.str.strip()  # 

# Preprocess the 'Ingredients' column
df['Ingredients'] = df['Ingredients'].apply(lambda x: [ingredient.strip(" '") for ingredient in x[1:-1].split(',')[:-1]])
df['Ingredients_str'] = df['Ingredients'].apply(lambda x: ' '.join(x))

# Use TfidfVectorizer to convert text data to numerical format for 'Ingredients'
vectorizer_ingredients = TfidfVectorizer(stop_words='english', min_df=1)
X_ingredients = vectorizer_ingredients.fit_transform(df['Ingredients_str'])

# Combine the text features with other features
X = pd.concat([df[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']], pd.DataFrame(X_ingredients.toarray())], axis=1)

# One-hot encode categorical columns 'Diet', 'Flavor', 'Carbohydrate_Content', and 'Protein'
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein']]).toarray()

# Concatenate the one-hot encoded columns with the rest of the features
X = pd.concat([X.drop(['Diet', 'Flavor', 'Carbohydrate_Content', 'Protein'], axis=1), pd.DataFrame(X_encoded)], axis=1)

# Cluster the data using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Initialize decision tree regressors for each target variable
regressors = {
    'Meal_Type': DecisionTreeRegressor(random_state=42),
    'Cuisine_Type': DecisionTreeRegressor(random_state=42),
    'Preparation_Method': DecisionTreeRegressor(random_state=42),
}

# Split the data into training and testing sets
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=42)

# Train decision tree regressors for each cluster
for cluster_id in df_train['Cluster'].unique():
    cluster_data = df_train[df_train['Cluster'] == cluster_id]
    for target, regressor in regressors.items():
        X_cluster = cluster_data.drop(['Cluster', 'Name', target], axis=1)  # Drop 'Name' column
        y_cluster = cluster_data[target]
        regressor.fit(X_cluster, y_cluster)

# Predictions on the test set
df_test['Predicted_Meal_Type'] = 0
df_test['Predicted_Cuisine_Type'] = 0
df_test['Predicted_Preparation_Method'] = 0

for index, row in df_test.iterrows():
    cluster_id = row['Cluster']
    for target, regressor in regressors.items():
        X_test_instance = row.drop(['Cluster', 'Name', target, 'Predicted_Meal_Type', 'Predicted_Cuisine_Type', 'Predicted_Preparation_Method'])
        predicted_value = regressor.predict([X_test_instance])[0]
        df_test.at[index, f'Predicted_{target}'] = predicted_value

# Evaluate the model
targets = ['Meal_Type', 'Cuisine_Type', 'Preparation_Method']
mse_values = {}

for target in targets:
    mse_values[target] = mean_squared_error(df_test[target], df_test[f'Predicted_{target}'])

print("Mean Squared Error for Each Target Variable:")
for target, mse in mse_values.items():
    print(f"{target}: {mse}")

# Display a few test predictions
test_predictions = df_test[['Meal_Type', 'Cuisine_Type', 'Preparation_Method', 'Predicted_Meal_Type', 'Predicted_Cuisine_Type', 'Predicted_Preparation_Method']].head()
print("\nTest Predictions:")
print(test_predictions)
