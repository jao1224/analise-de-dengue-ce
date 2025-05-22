import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from datetime import datetime
import plotly.io as pio

# Configurar para salvar os gráficos como arquivos estáticos
pio.renderers.default = "png"

# Carregar o conjunto de dados de dengue
df = pd.read_csv('dengue_ce_tratado.csv')

# Converter data para datetime se existir
if 'data_iniSE' in df.columns:
    df['data_iniSE'] = pd.to_datetime(df['data_iniSE'])
    df['ano'] = df['data_iniSE'].dt.year
    df['mes'] = df['data_iniSE'].dt.month
else:
    # Se não tiver coluna de data, criar uma baseada no índice (para demonstração)
    hoje = datetime.now()
    df['data_iniSE'] = pd.date_range(end=hoje, periods=len(df), freq='M')
    df['ano'] = df['data_iniSE'].dt.year
    df['mes'] = df['data_iniSE'].dt.month

# Exibir informações sobre o conjunto de dados
print(df.info())

# Selecionar variáveis explicativas
variaveis = ['pop', 'tempmin', 'umidmax', 'tempmed', 'tempmax', 'umidmed', 'umidmin', 'p_inc100k', 'Rt']
X = df[variaveis]

# Selecionar a variável alvo
y = df['casos']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar e treinar o modelo DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
previsoes = dt.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print("Métricas de Desempenho:")
print(f"MAE  (Erro Absoluto Médio):       {mae:.2f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

# Criar DataFrame com métricas para o gráfico de barras
metricas_df = pd.DataFrame({
    'Métrica': ['MAE', 'R²'],
    'Valor': [mae, r2]
})

# Criar gráfico de barras das métricas
fig_metricas = px.bar(metricas_df, x='Métrica', y='Valor',
                      title='Métricas de Desempenho do Modelo',
                      labels={'Valor': 'Valor da Métrica'},
                      color='Métrica')

# Salvar gráfico de métricas como arquivo estático
pio.write_image(fig_metricas, 'metricas_desempenho.png', scale=2, width=800, height=500)
print("Gráfico de métricas salvo como 'metricas_desempenho.png'")

# Criar gráfico de dispersão (valores reais vs. previstos)
fig_dispersao = px.scatter(x=y_test, y=previsoes,
                          title='Valores Reais vs. Previstos',
                          labels={'x': 'Valores Reais', 'y': 'Valores Previstos'})

# Adicionar linha de referência
fig_dispersao.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Linha de Referência'
))

# Criar texto com as métricas
texto_metricas = f'Métricas de Desempenho:<br>'
texto_metricas += f'MAE: {mae:.2f}<br>'
texto_metricas += f'R²: {r2:.4f}'

# Adicionar anotação com as métricas
fig_dispersao.add_annotation(
    x=0.02,
    y=0.98,
    xref='paper',
    yref='paper',
    text=texto_metricas,
    showarrow=False,
    font=dict(size=12),
    align='left',
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='rgba(0, 0, 0, 0.3)',
    borderwidth=1,
    borderpad=4
)

# Salvar gráfico de dispersão como arquivo estático
pio.write_image(fig_dispersao, 'dispersao_real_vs_previsto.png', scale=2, width=800, height=500)
print("Gráfico de dispersão salvo como 'dispersao_real_vs_previsto.png'")

# PREVISÃO PARA OS PRÓXIMOS 6 MESES
# Obter a última data no conjunto de dados
ultima_data = df['data_iniSE'].max()

# Criar datas para os próximos 6 meses
datas_futuras = [ultima_data + pd.DateOffset(months=i+1) for i in range(6)]
meses_futuros = [data.month for data in datas_futuras]
anos_futuros = [data.year for data in datas_futuras]

# Criar DataFrame para previsões futuras com base em padrões sazonais
futuros_df = pd.DataFrame()

for i, (mes, ano) in enumerate(zip(meses_futuros, anos_futuros)):
    # Filtrar dados históricos do mesmo mês (para capturar sazonalidade)
    dados_mes_similar = df[df['mes'] == mes]
    
    if len(dados_mes_similar) > 0:
        # Se temos dados históricos para este mês, usar a média dos últimos 3 anos
        dados_recentes = dados_mes_similar.sort_values('ano', ascending=False).head(3)
        valores_medios = dados_recentes[variaveis].mean().to_dict()
    else:
        # Se não temos dados históricos para este mês, usar a média dos últimos 6 meses
        valores_medios = df.tail(6)[variaveis].mean().to_dict()
    
    # Adicionar ao DataFrame de previsões futuras
    futuros_df = pd.concat([futuros_df, pd.DataFrame([valores_medios])], ignore_index=True)

# Prever os casos para os próximos 6 meses
previsoes_futuras = dt.predict(futuros_df)

# Criar DataFrame com resultados para melhor visualização
resultados_futuros = pd.DataFrame({
    'Data': datas_futuras,
    'Mês': meses_futuros,
    'Ano': anos_futuros,
    'Previsão de Casos': previsoes_futuras.round().astype(int)
})

print('Previsão para os próximos 6 meses:')
print(resultados_futuros[['Data', 'Previsão de Casos']])

# Criar gráfico de previsão para os próximos 6 meses
fig_previsao = go.Figure()

# Adicionar dados históricos (últimos 12 meses, se disponíveis)
if 'data_iniSE' in df.columns:
    df_historico = df.sort_values('data_iniSE').tail(12)
    fig_previsao.add_trace(go.Scatter(
        x=df_historico['data_iniSE'],
        y=df_historico['casos'],
        mode='lines+markers',
        name='Casos Históricos',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

# Adicionar previsões futuras
fig_previsao.add_trace(go.Scatter(
    x=resultados_futuros['Data'],
    y=resultados_futuros['Previsão de Casos'],
    mode='lines+markers',
    name='Previsão de Casos',
    line=dict(color='green', width=2),
    marker=dict(size=10, symbol='diamond')
))

# Adicionar área sombreada para indicar incerteza nas previsões
fig_previsao.add_trace(go.Scatter(
    x=resultados_futuros['Data'].tolist() + resultados_futuros['Data'].tolist()[::-1],
    y=(resultados_futuros['Previsão de Casos'] * 1.2).tolist() + 
      (resultados_futuros['Previsão de Casos'] * 0.8).tolist()[::-1],
    fill='toself',
    fillcolor='rgba(0,128,0,0.2)',
    line=dict(color='rgba(0,128,0,0)'),
    hoverinfo='skip',
    showlegend=False
))

# Personalizar layout
fig_previsao.update_layout(
    title='Previsão de Casos de Dengue para os Próximos 6 Meses',
    xaxis_title='Data',
    yaxis_title='Número de Casos',
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.3)',
        borderwidth=1
    ),
    hovermode='x unified',
    plot_bgcolor='white',
    xaxis=dict(
        tickformat='%b/%Y',
        tickangle=-45,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        gridcolor='lightgray'
    )
)

# Adicionar anotações para os valores previstos
for i, row in resultados_futuros.iterrows():
    fig_previsao.add_annotation(
        x=row['Data'],
        y=row['Previsão de Casos'],
        text=f"{row['Previsão de Casos']}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30
    )

# Salvar gráfico de previsão como arquivo estático
pio.write_image(fig_previsao, 'previsao_6meses_dengue.png', scale=2, width=1000, height=600)
print("Gráfico de previsão para 6 meses salvo como 'previsao_6meses_dengue.png'")

# Salvar também em formato HTML para interatividade (opcional)
pio.write_html(fig_previsao, 'previsao_6meses_dengue.html')
print("Versão interativa do gráfico salva como 'previsao_6meses_dengue.html'")