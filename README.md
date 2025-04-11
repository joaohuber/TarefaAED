# 🚢 Análise e Tratamento do Dataset Titanic

Este projeto realiza uma análise completa do dataset **Titanic**, utilizando as bibliotecas Python `pandas`, `seaborn` e `matplotlib`. O objetivo é limpar, tratar e explorar os dados de forma visual e estatística.

---

## 📥 1. Carregamento dos Dados

Os dados são carregados diretamente da biblioteca `seaborn`, que possui uma versão pré-processada do dataset Titanic.

```python
df = sns.load_dataset('titanic')
```

---

## 👀 2. Visualização Inicial

A estrutura inicial do dataset é visualizada com `df.head()`, permitindo entender os tipos de dados e suas colunas principais.

---

## 🔍 3. Verificação de Dados Faltantes

É feita uma análise de dados ausentes em cada coluna com o método:

```python
df.isnull().sum()
```

---

## 🧼 4. Tratamento de Dados

### ✏️ 4.1 Preenchimento de Valores Faltantes

- `age` e `fare`: preenchidos com a **mediana** por serem numéricos e conterem outliers.
- `embarked`: preenchido com a **moda**, pois é categórico.

```python
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
```

### 🧹 4.2 Remoção de Outliers (IQR)

Utiliza o Intervalo Interquartil para remover outliers nas colunas `age` e `fare`.

```python
def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]

df = remover_outliers(df, 'age')
df = remover_outliers(df, 'fare')
```

---

## ✅ 5. Verificação Pós-Tratamento

Confirmação de que não há mais valores ausentes após o tratamento.

```python
df.isnull().sum()
```

---

## 📊 6. Estatísticas Descritivas

As estatísticas descritivas ajudam a entender melhor a distribuição e variabilidade dos dados:

```python
df.describe(include='all')
```

---

## 📈 7. Visualizações Exploratórias

### 📌 7.1 Histograma – Distribuição de Idade

Mostra a distribuição de idade com curva de densidade.

```python
sns.histplot(df['age'].dropna(), kde=True, bins=30)
```

### 🧳 7.2 Boxplot – Idade por Classe

Relaciona idade dos passageiros com suas respectivas classes (`pclass`).

```python
sns.boxplot(x='pclass', y='age', data=df)
```

### 🚻 7.3 Gráfico de Barras – Sobrevivência por Sexo

Compara a taxa de sobrevivência entre homens e mulheres.

```python
sns.countplot(x='sex', hue='survived', data=df)
```

### 💸 7.4 Scatter Plot – Relação entre Idade e Tarifa

Mostra a correlação entre idade e tarifa paga, com cores indicando sobrevivência.

```python
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
```

---

## 💾 8. Exportação dos Dados

O dataset tratado é exportado para um arquivo `.csv`:

```python
df.to_csv('titanic_tratado.csv', index=False)
```

---

## 🧠 Conclusão

Com esse processo, temos um dataset limpo e pronto para análises mais profundas, como modelos de machine learning ou relatórios com dashboards. A remoção de outliers e o tratamento adequado dos dados garantem maior confiabilidade nos resultados.

---

## 🛠 Tecnologias Utilizadas

- Python
- Pandas
- Seaborn
- Matplotlib
