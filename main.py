import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Etapa 1: Carregar o dataset Titanic
df = sns.load_dataset('titanic')

# Etapa 2: Visualizar as primeiras linhas
print("Visualização inicial:")
print(df.head())

# Etapa 3: Verificar dados faltantes
print("\nDados faltantes:")
print(df.isnull().sum())

# ==================== ETAPA NOVA: TRATAMENTO ======================

# --- 1. Preenchendo valores faltantes ---

# Preencher idade com a mediana (por ter outliers)
df['age'] = df['age'].fillna(df['age'].median())

# Preencher tarifa com a mediana
df['fare'] = df['fare'].fillna(df['fare'].median())

# Preencher porto de embarque com a moda (valor mais comum)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# --- 2. Removendo outliers de 'age' e 'fare' usando IQR ---
def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Filtrar os dados dentro dos limites
    return df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]

df = remover_outliers(df, 'age')
df = remover_outliers(df, 'fare')

# Conferindo se ainda tem valores faltantes
print("\nApós tratamento, dados faltantes:")
print(df.isnull().sum())

# Etapa 4: Estatística descritivas
print("\nEstatísticas descritivas:")
print(df.describe(include='all'))

# Etapa 5: Histograma - distribuição de idade
plt.figure(figsize=(10, 5))
sns.histplot(df['age'].dropna(), kde=True, bins=30, color='skyblue')
plt.title('Distribuição de Idade Passageiros')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Etapa 6: Boxplot - Idade por classe
plt.figure(figsize=(10,5))
sns.boxplot(x='pclass', y='age', data=df, palette='pastel')
plt.title('Idade por Classe de Passagem')
plt.xlabel('Classe')
plt.ylabel('Idade')
plt.show()

# Etapa 7: Gráfico de barras - Sobrevivência por sexo
plt.figure(figsize=(10, 5))
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title('Sobrevivência por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Número de Passageiros')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.show()

# Etapa 8: Scatter plot - Idade vs Tarifa
plt.figure(figsize=(10, 5))
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title('Relação entre Idade e Tarifa')
plt.xlabel('Idade')
plt.ylabel('Tarifa')

plt.show()

#fazendo o load dos dados já tratados
df.to_csv('titanic_tratado.csv', index=False)
