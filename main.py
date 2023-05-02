import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

""" 1. Načtení souboru"""
df = pd.read_csv('data/Car_sales.csv')

""" 2. výpis prvních pěti řádků datasetu """
print(df.head())

""" 3. statistické informace o datasetu """
print(df.describe())

""" 4. počet chybějících hodnot v každém sloupci """
print(df.isnull().sum())

""" 5. Vypsání každého řádku datasetu """
for index, row in df.iterrows():
    print(row)
    print("\n")

""" 6. Vztah mezi výkonem motoru a objemem motoru"""
sns.scatterplot(x="Engine_size", y="Horsepower", data=df)
plt.title("Vztah mezi výkonem a objemem motoru"
          "")
plt.show()

""" 7. Vztah mezi výkonem motoru a objemem motoru s predikovanými hodnotami"""
X = np.array(df["Engine_size"]).reshape(-1, 1)
Y = np.array(df["Horsepower"])

model = LinearRegression()
print(model.fit(X, Y))

# Predikce výstupů na základě vstupních dat
y_pred = model.predict(X)

# Vykreslení originálních hodnot a predikovaných hodnot
plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)

# Nastavení popisků os a titulku grafu
plt.xlabel("Engine size")
plt.ylabel("Horsepower")
plt.title("Engine size vs. Horsepower")

# Zobrazení grafu
plt.show()