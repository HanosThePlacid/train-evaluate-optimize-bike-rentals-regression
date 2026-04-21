# Bike Rentals Regression Models

Microsoft Learn: Regression module gyakorlati kódjai (daily-bike-share.csv adatbázison).

## Tanultam a modulban

- Mi a regresszió és mikor használjuk
- Adatok felfedezése (histogramm, boxplot, scatter plot, korreláció)
- Train/test split
- Egyszerű lineáris regresszió
- További modellek: Lasso, Decision Tree, Random Forest, Gradient Boosting
- Hyperparameter tuning GridSearchCV-vel
- Pipeline + preprocessing (StandardScaler + OneHotEncoder)
- Modell mentése és betöltése (joblib)
- Értékelés: MSE, RMSE, R²
- Predikció új adatokra

## Fájlok és mit csinálnak

| Fájl | Modell(ek) | Mit csinál |
|------|------------|----------|
| `linear_regression.py` | LinearRegression | Alap lineáris regresszió + kiértékelés + plot |
| `regression_models.py` | Lasso, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor | Több modell kipróbálása és összehasonlítása |
| `optimize_regression.py` | GradientBoostingRegressor + GridSearchCV + Pipeline (RandomForest) | Hyperparameter optimalizálás + preprocessing pipeline + modell mentése |
| `linear_regression_data_eval.py` | - | Adatvizualizáció (hisztogramok, boxplotok, scatter plotok, korreláció) |

## Requirements
```bash
pip install -r requirements.txt