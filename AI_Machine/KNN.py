import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Function to run KNN for specified genres
def run_knn_for_genres(genres):
    # Load the dataset
    data = pd.read_csv('AnimeList.csv')

    print("Dataset preview:")
    print(data.head())
    print("Available columns:", data.columns)

    # Initialize storage for results
    results = {}

    for genre in genres:
        # Filter for the specified genre
        genre_data = data[data['genre'].str.contains(genre, na=False)].copy()

        if genre_data.empty:
            print(f"No anime found for the genre: {genre}")
            continue

        # Preprocessing the data
        label_encoder = LabelEncoder()
        categorical_cols = ['type', 'source',]

        for col in categorical_cols:
            if col in genre_data.columns:
                genre_data[col] = label_encoder.fit_transform(genre_data[col].fillna('Unknown'))
            else:
                print(f"Column '{col}' not found. Please check the column names.")

        genre_data = genre_data.drop(['title', 'anime_id'], axis=1, errors='ignore')

        if 'score' in genre_data.columns:
            X = genre_data.drop('score', axis=1)
            y = genre_data['score'].fillna(genre_data['score'].mean())

            X = X.select_dtypes(include=[np.number])
            if X.shape[1] == 0:
                print("No numeric features found for training. Please check your dataset.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Set neighbors and ensure it does not exceed the number of training samples
                max_neighbors = min(len(X_train), 20)  # Adjust max_neighbors as needed
                neighbors = np.arange(1, max_neighbors + 1)

                train_accuracy = []
                test_accuracy = []
                cv_scores = []

                for k in neighbors:
                    knn_model = KNeighborsRegressor(n_neighbors=k)
                    knn_model.fit(X_train, y_train)

                    train_accuracy.append(knn_model.score(X_train, y_train))
                    test_accuracy.append(knn_model.score(X_test, y_test))

                    # Perform 10-fold cross-validation
                    cv_score = cross_val_score(knn_model, X, y, cv=10, scoring='neg_mean_squared_error')
                    cv_scores.append(np.sqrt(-cv_score.mean()))  # Store RMSE

                results[genre] = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'cv_scores': cv_scores,
                    'neighbors': neighbors,
                    'y_test': y_test,
                    'X_test': X_test
                }

    # Plot accuracy and RMSE for combined genres using plotly
    if results:
        fig = go.Figure()

        for genre, result in results.items():
            fig.add_trace(go.Scatter(x=result['neighbors'], y=result['train_accuracy'], mode='lines+markers', name=f'Training Accuracy ({genre})'))
            fig.add_trace(go.Scatter(x=result['neighbors'], y=result['test_accuracy'], mode='lines+markers', name=f'Testing Accuracy ({genre})'))
            fig.add_trace(go.Scatter(x=result['neighbors'], y=result['cv_scores'], mode='lines+markers', name=f'10-Fold CV RMSE ({genre})'))

        fig.update_layout(title="Accuracy and RMSE vs n_neighbors for Multiple Genres (KNN)",
                          xaxis_title="n_neighbors",
                          yaxis_title="Accuracy / RMSE",
                          template="plotly_white")
        fig.show()

    # Plot actual vs predicted for each genre using plotly
    fig_actual_vs_predicted = go.Figure()

    for genre, result in results.items():
        knn_model = KNeighborsRegressor(n_neighbors=min(5, max(result['neighbors'])))  # Use the best value for neighbors
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(result['X_test'])

        fig_actual_vs_predicted.add_trace(go.Scatter(x=result['y_test'], y=y_pred, mode='markers', name=f'Actual vs Predicted ({genre})'))

    fig_actual_vs_predicted.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='Ideal Prediction Line', line=dict(color='red')))
    fig_actual_vs_predicted.update_layout(title="Actual vs Predicted Anime Scores for Multiple Genres (KNN)",
                                          xaxis_title="Actual Score",
                                          yaxis_title="Predicted Score",
                                          template="plotly_white")
    fig_actual_vs_predicted.show()

    # Summary of results
    for genre, result in results.items():
        knn_model = KNeighborsRegressor(n_neighbors=min(5, max(result['neighbors'])))
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(result['X_test'])
        mae = mean_absolute_error(result['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(result['y_test'], y_pred))

        print(f"Results for {genre}: MAE = {mae}, RMSE = {rmse}")

run_knn_for_genres(['Romance', 'Kids', 'Demons'])
