import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar o conjunto de dados de dengue
df = pd.read_csv('dengue_ce_tratado.csv')

# Selecionar variáveis explicativas
variaveis = ['pop', 'tempmin', 'umidmax', 'tempmed', 'tempmax', 'umidmed', 'umidmin', 'p_inc100k', 'Rt']
X = df[variaveis]

# Selecionar a variável alvo
y = df['casos']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados (importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar e treinar o modelo KNN (K=5 é um valor comum)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
previsoes = knn.predict(X_test_scaled)

# Avaliação do modelo e visualização
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, previsoes)

print("Métricas de Desempenho:")
print(f"MAE  (Erro Absoluto Médio):       {mae:.2f}")
print(f"MSE  (Erro Quadrático Médio):     {mse:.2f}")
print(f"RMSE (Raiz do Erro Quadrático):   {rmse:.2f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

# Criar gráfico de dispersão (valores reais vs. previstos)
fig_dispersao = px.scatter(x=y_test, y=previsoes,
                          title='Valores Reais vs. Previstos (KNN)',
                          labels={'x': 'Valores Reais', 'y': 'Valores Previstos'})

# Adicionar uma única linha de tendência
fig_dispersao.add_traces(px.scatter(x=y_test, y=previsoes, trendline='ols').data[1])

# Adicionar anotação com as métricas
fig_dispersao.add_annotation(
    x=0.02,
    y=0.98,
    xref='paper',
    yref='paper',
    text=f"MAE: {mae:.2f}<br>MSE: {mse:.2f}<br>RMSE: {rmse:.2f}<br>R²: {r2:.4f}",
    showarrow=False,
    font=dict(size=12),
    align='left',
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='rgba(0, 0, 0, 0.3)',
    borderwidth=1,
    borderpad=4
)

fig_dispersao.show()