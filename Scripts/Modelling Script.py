import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np

# Load the Excel file
xls = pd.ExcelFile('Training.xlsx')

# Define base models and their parameter grids
param_grids = {
    'linear_regression': {},
    'decision_tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'random_forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    'gradient_boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]},
    'svr': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
}

base_models = [
    ('linear_regression', LinearRegression()),
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(random_state=42)),
    ('svr', SVR())
]

# Generate all permutations of pairs of models
model_combinations = list(combinations(base_models, 2))

# Add individual base models to combinations
model_combinations.extend([(model,) for model in base_models])

# Define meta-model
meta_model = RandomForestRegressor(random_state=42)

# Prepare to save results to Excel
with pd.ExcelWriter('Model_Evaluation.xlsx', engine='openpyxl') as writer:
    # Iterate through each sheet and process
    results = {}
    feature_importance_combined = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Drop the Movement_Category column
        if 'Movement_Category' in df.columns:
            df = df.drop(columns=['Movement_Category','Date'])

        # Impute missing values
        df.fillna({'Theme': 'None'}, inplace=True)  # For Theme column
        df.fillna(0, inplace=True)  # For numerical columns

        # Convert Theme column to numerical values using one-hot encoding
        df = pd.get_dummies(df, columns=['Theme'], drop_first=True)

        # Define the target and features
        X = df.drop(columns=['Zscore_Movement'])
        y = df['Zscore_Movement']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate each combination of models without cross-validation
        sheet_results = {}
        feature_importances_combined = []
        for combo in model_combinations:
            try:
                # Tune each base model using RandomizedSearchCV
                tuned_estimators = []
                best_params = {}
                for name, model in combo:
                    param_grid = param_grids[name]
                    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
                    random_search.fit(X_train, y_train)
                    best_model = random_search.best_estimator_
                    tuned_estimators.append((name, best_model))
                    best_params[name] = random_search.best_params_

                stacking_regressor = StackingRegressor(estimators=tuned_estimators, final_estimator=meta_model)
                stacking_regressor.fit(X_train, y_train)

                y_pred = stacking_regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)  # Updated function
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                model_names = ' + '.join([name for name, _ in combo])
                sheet_results[model_names] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R-squared': r2,
                    'Best Params': best_params
                }

                # Get feature importances for each base model
                for name, model in tuned_estimators:
                    if hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        feature_importances_df = pd.DataFrame({
                            'Model': name,
                            'Feature': X.columns,
                            'Importance': feature_importances,
                            'Ensemble': model_names  # Flagging the ensemble name
                        }).sort_values(by='Importance', ascending=False)
                        feature_importances_combined.append(feature_importances_df)

            except Exception as e:
                model_names = ' + '.join([name for name, _ in combo])
                print(f'Error with model combination: {model_names}')
                print(e)

        results[sheet_name] = sheet_results

        # Combine feature importances from all models
        if feature_importances_combined:
            combined_importances_df = pd.concat(feature_importances_combined)

        # Save combined feature importances to Excel
        combined_importances_df.to_excel(writer, sheet_name=sheet_name + '_Feature_Importance')

    # Save results to Excel
    for sheet_name, sheet_results in results.items():
        df_results = pd.DataFrame(sheet_results).T
        df_results['Best Params'] = df_results['Best Params'].apply(lambda x: str(x))
        df_results.to_excel(writer, sheet_name=sheet_name + '_Metrics')
