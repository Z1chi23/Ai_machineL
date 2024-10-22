import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Function to run Random Forest for specified genres
def run_random_forest_for_genres(genres):
    # Load the dataset
    data = pd.read_csv('AnimeList.csv')

    # Initialize storage
    results = {}

    # Variables to track global min and max values for y_test and y_pred
    global_min_y = float('inf')
    global_max_y = float('-inf')

    for genre in genres:
        # Filter the specified genre
        genre_anime = data[data['genre'].str.contains(genre, na=False)].copy()

        if genre_anime.empty:
            print(f"No anime found for the genre: {genre}")
            continue

        # Label encoding for category
        label_encoder = LabelEncoder()
        categorical_cols = ['type', 'source']

        for col in categorical_cols:
            if col in genre_anime.columns:
                genre_anime.loc[:, col] = label_encoder.fit_transform(genre_anime[col].fillna('Unknown'))

        # Prepare target variable
        X = genre_anime.drop(columns=['anime_id', 'title', 'score', 'genre'], errors='ignore')
        y = genre_anime['score']

        # Check for NaN values in X and y
        if X.isnull().values.any() or y.isnull().values.any():
            # Fill NaN values with the mean for numerical columns in X
            X = X.fillna(X.mean())
            # Fill NaN values in y with the mean of y
            y = y.fillna(y.mean())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Model
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best model
        best_rf_model = grid_search.best_estimator_

        # 10-Fold Cross Validation
        cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring='neg_mean_squared_error')
        mean_cv_rmse = np.sqrt(-cv_scores.mean())

        # Predictions
        y_pred = best_rf_model.predict(X_test)

        # Update global min and max values for y_test and y_pred
        global_min_y = min(global_min_y, min(y_test), min(y_pred))
        global_max_y = max(global_max_y, max(y_test), max(y_pred))

        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Genre: {genre}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"10-Fold CV Mean RMSE: {mean_cv_rmse}")

        # Store results for later comparison
        results[genre] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse,
            'cv_rmse': mean_cv_rmse,
            'model': best_rf_model  # Store the trained model
        }

    # Plotting Actual vs Predicted for all genres using Plotly
    if results:
        fig = go.Figure()

        for genre, result in results.items():
            fig.add_trace(go.Scatter(x=result['y_test'], y=result['y_pred'], mode='markers', name=f'{genre}'))

        # Add prediction line based on global min and max values
        fig.add_trace(go.Scatter(x=[global_min_y, global_max_y], y=[global_min_y, global_max_y], mode='lines', name='Ideal Prediction Line', line=dict(color='red')))
        fig.update_layout(title="Actual vs Predicted Anime Scores for Multiple Genres (Random Forest)",
                          xaxis_title="Actual Score",
                          yaxis_title="Predicted Score",
                          template="plotly_white")
        fig.show()

        # Histogram using Plotly for all genres
        fig_histogram = go.Figure()

        for genre, result in results.items():
            fig_histogram.add_trace(go.Histogram(x=result['y_test'], name=f'Actual Scores - {genre}', opacity=0.5, marker_color='blue'))
            fig_histogram.add_trace(go.Histogram(x=result['y_pred'], name=f'Predicted Scores - {genre}', opacity=0.5, marker_color='orange'))

        fig_histogram.update_layout(barmode='overlay', title="Histogram of Actual vs Predicted Scores for Multiple Genres",
                                    xaxis_title="Scores", yaxis_title="Frequency", template="plotly_white")
        fig_histogram.show()

        # Feature Importance using Plotly
        first_genre = next(iter(results))
        best_rf_model = results[first_genre]['model']  # Get the model from the first genre
        feature_importances = best_rf_model.feature_importances_
        features = X.columns

        fig_importance = px.bar(x=features, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'}, title='Feature Importance in Random Forest Model (Example Genre)', template='plotly_white')
        fig_importance.show()

run_random_forest_for_genres(['Romance', 'Kids', 'Josei', 'Magic'])
