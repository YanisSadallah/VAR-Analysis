# Import des bibliothèques 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
import statsmodels.api as sm
# Outils pour tests de stationnarité et modèles VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
# Pour télécharger les données financières
import yfinance as yf 
# Pour évaluer la qualité des prévisions
from sklearn.metrics import mean_squared_error




#1 - Téléchargement des données 
# On télécharge les données mensuelles de BNP Paribas et de l'ETF VGK entre 2014 et 2019
bnp = yf.download('BNP.PA', start='2014-01-01', end='2019-12-31', interval='1mo')
vgk = yf.download('VGK', start='2014-01-01', end='2019-12-31', interval='1mo')
# On concatène les colonnes Close des deux séries en les alignant sur leurs dates communes
data = pd.concat([bnp['Close'], vgk['Close']], axis=1)
data.columns = ['BNP', 'VGK']
data = data.dropna()


# 2. Transformation logarithmique et calcul des rendements
log_data = np.log(data)
# On calcule les rendements log : différence du log
returns = log_data.diff().dropna()

# 3. Test de stationnarité ADF
# Fonction pour effectuer le test ADF et afficher les résultats
def run_adf(series, name):
    result = adfuller(series)
    print(f"\nADF Test pour {name}")
    print(f"Statistique ADF : {result[0]}")
    print(f"p-value : {result[1]}")
    print("Conclusion :", "Stationnaire" if result[1] < 0.05 else "Non stationnaire")

# Application du test ADF sur les rendements
run_adf(returns['BNP'], "BNP")
run_adf(returns['VGK'], "VGK")



# ================================
# 4. Estimation du modèle VAR
# ================================

# On crée un modèle VAR à partir des rendements log
model = VAR(returns)

# Estimation du modèle avec 2 retards (ordre p=2, comme dans ton rapport)
results = model.fit(2)

# Affichage du résumé du modèle VAR
print("\nRésumé du modèle VAR :")
print(results.summary())

# ================================
# 5. Prévision à horizon 2 périodes
# ================================

# On utilise les 2 dernières observations pour faire les prévisions
forecast = results.forecast(returns.values[-2:], steps=2)

# On stocke les résultats dans un DataFrame pour lisibilité
forecast_df = pd.DataFrame(forecast, columns=['BNP_forecast', 'VGK_forecast'])
print("\nPrévision des rendements log à 2 pas :")
print(forecast_df)

# ================================
# 6. Conversion en niveaux de prix
# ================================

# Derniers prix observés
last_prices = data.iloc[-1]

# On cumule les rendements prévus pour obtenir le log-prix
predicted_log_returns = forecast_df.cumsum()

# On reconvertit en prix (exponentielle du log)
predicted_prices = last_prices.values * np.exp(predicted_log_returns)

# Affichage des prix prévus
print("\nPrix prévus :")
print(pd.DataFrame(predicted_prices, columns=data.columns))

# ================================
# 7. (Optionnel) Évaluation avec RMSE
# ================================

# Si tu as les vraies valeurs futures, tu peux comparer ici
# rmse_bnp = mean_squared_error(y_true, y_pred_bnp, squared=False)
# rmse_vgk = mean_squared_error(y_true, y_pred_vgk, squared=False)

# ================================
# 8. Fonctions de réponse impulsionnelle (IRF)
# ================================

# On calcule les IRF sur 10 périodes
irf = results.irf(10)

# Affichage des IRFs
irf.plot(orth=False)
plt.tight_layout()
plt.show()
