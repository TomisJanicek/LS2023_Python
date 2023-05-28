import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QComboBox, QMessageBox, \
    QDialog, QTextEdit, QGridLayout

df = pd.read_csv('data/Car_sales.csv')

class AnalyzeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Zpracování CSV dat')


        # Vytvoření komponent
        self.An1_button = QPushButton('Výpis prvních pěti řádků datasetu')
        self.An1_button.clicked.connect(self.first_five)

        self.An2_button = QPushButton('Statistické informace o datasetu')
        self.An2_button.clicked.connect(self.statsOfCSV)

        self.An_clear_button = QPushButton('Zmizík')
        self.An_clear_button.clicked.connect(self.clearIT)

        self.data_textedit = QTextEdit()
        self.data_textedit.setReadOnly(True)


        #self.label1 = QLabel('První číslo:') --- label
        #self.input1 = QLineEdit() --- input

        #self.result_label = QLabel('Výsledek:')
        #self.result = QLabel()

        #self.comboBox_label = QLabel('Možnosti:')
        #self.comboBox = QComboBox()
        #self.comboBox.addItem("A")
        #self.comboBox.addItem("B")
        #self.comboBox.addItem("C")

        #self.print_button = QPushButton('Vypsat')
        #self.print_button.clicked.connect(self.print_selection)

        # Vytvoření rozložení
        layout = QGridLayout()

        layout.addWidget(self.An1_button,0,0)
        layout.addWidget(self.An2_button,0,1)
        layout.addWidget(self.An_clear_button,3,0,1,2) #Význam čísel: řádek, sloupec,sloupec,počet sloupců

        """
        layout.addWidget(self.label1)
        layout.addWidget(self.input1)
        layout.addWidget(self.label2)
        layout.addWidget(self.input2)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.comboBox_label)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.print_button)
        """
        layout.addWidget(self.data_textedit, 4, 0, 1, 2)


        self.setLayout(layout)
        self.showMaximized()

    def calculate_sum(self):
        try:
            num1 = float(self.input1.text())
            num2 = float(self.input2.text())
            sum = num1 + num2
            self.result.setText(str(sum))
        except ValueError:
            self.result.setText('Neplatný vstup')

    def print_selection(self):
        selected_option = self.comboBox.currentText()
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Vybráno:")
        msg_box.setIcon(QMessageBox.Icon.Information)  # Odstranění ikony
        msg_box.setStandardButtons(QMessageBox.StandardButton.Close)  # Nastavení prázdných tlačítek
        msg_box.setText(f'Vybraná možnost: {selected_option}')
        msg_box.exec()

    def first_five(self): #1. Výpis prvních pěti řádků datasetu
        text = df.head().to_string()
        self.data_textedit.setPlainText(text)

    def statsOfCSV(self):
        text = df.describe().to_string()
        self.data_textedit.setPlainText(text)

    def clearIT(self):
        text = ""
        self.data_textedit.setPlainText(text)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    analyze = AnalyzeApp()
    analyze.show()
    sys.exit(app.exec())


"""
""" #4. počet chybějících hodnot v každém sloupci
"""
print(" 4. počet chybějících hodnot v každém sloupci ")
print(df.isnull().sum())

""" #5. Vypsání každého řádku datasetu
"""
print(" 5. Vypsání každého řádku datasetu ")
for index, row in df.iterrows():
    print(row)
    print("\n")

""" #6. Vztah mezi výkonem motoru a objemem motoru
"""
sns.scatterplot(x="Engine_size", y="Horsepower", data=df)
plt.title("Vztah mezi výkonem a objemem motoru")
plt.show()

""" #7. Vztah mezi výkonem motoru a objemem motoru s predikovanými hodnotami
"""
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
plt.title("Vztah mezi výkonem motoru a objemem motoru \ns predikovanými hodnotami")

# Zobrazení grafu
plt.show()

"""