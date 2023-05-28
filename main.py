import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.linear_model import LinearRegression


import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, \
    QTextEdit, QGridLayout

df = pd.read_csv('data/Car_sales.csv')


class AnalyzeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Zpracování CSV dat')

        # Vytvoření komponent
        self.An1_button = QPushButton('Výpis prvních pěti řádků datasetu')
        self.An1_button.clicked.connect(self.first_five)

        self.An2_button = QPushButton('Statistické informace o datasetu')
        self.An2_button.clicked.connect(self.stats_of_csv)

        self.An3_button = QPushButton('počet chybějících hodnot v každém sloupci')
        self.An3_button.clicked.connect(self.numb_of_miss_value)

        self.An4_button = QPushButton('Vypsání každého řádku datasetu')
        self.An4_button.clicked.connect(self.print_all)

        self.An_clear_button = QPushButton('Zmizík')
        self.An_clear_button.clicked.connect(self.clear_it)

        self.Save_button = QPushButton('Uložit')
        self.Save_button.clicked.connect(self.Save)

        self.Save_button2 = QPushButton('Uložit')
        self.Save_button.clicked.connect(self.Save)

        self.An5_button = QPushButton('Vztah mezi:')
        self.An5_button.clicked.connect(self.vztah)

        self.An6_button = QPushButton('Vztah mezi (s predikcí hondot):')
        self.An6_button.clicked.connect(self.vztah_s_predikci)

        self.combo_box1 = QLabel('Možnosti:')
        self.combo_box1 = QComboBox(self)
        self.combo_box2 = QLabel('Možnosti:')
        self.combo_box2 = QComboBox(self)

        self.combo_box3 = QLabel('Možnosti:')
        self.combo_box3 = QComboBox(self)
        self.combo_box4 = QLabel('Možnosti:')
        self.combo_box4 = QComboBox(self)

        self.data_textedit = QTextEdit()
        self.data_textedit.setReadOnly(True)

        self.label = QLabel("Prvně ukaž vztah, poté ulož")
        self.label2 = QLabel("Prvně ukaž vztah, poté ulož")

        # Vytvoření rozložení
        layout = QGridLayout()

        layout.addWidget(self.An1_button, 0, 0)
        layout.addWidget(self.An2_button, 0, 1)
        layout.addWidget(self.An3_button, 1, 0)
        layout.addWidget(self.An4_button, 1, 1)
        layout.addWidget(self.An5_button, 2, 0, 1, 1)
        layout.addWidget(self.combo_box1, 2, 1, 1, 1)
        layout.addWidget(self.combo_box2, 2, 2, 1, 1)
        layout.addWidget(self.Save_button, 2, 3, 1, 1)
        layout.addWidget(self.label, 2, 4, 1, 1)
        layout.addWidget(self.An6_button, 3, 0, 1, 1)
        layout.addWidget(self.combo_box3, 3, 1, 1, 1)
        layout.addWidget(self.combo_box4, 3, 2, 1, 1)
        layout.addWidget(self.Save_button2, 3, 3, 1, 1)
        layout.addWidget(self.label2, 3, 4, 1, 1)
        layout.addWidget(self.An_clear_button, 4, 0, 1, 5)
        layout.addWidget(self.data_textedit, 5, 0, 1, 5)

        self.setLayout(layout)
        self.showMaximized()
        self.load_data()

    def first_five(self): # 1. Výpis prvních pěti řádků datasetu
        text = df.head().to_string()
        self.data_textedit.setPlainText(text)

    def stats_of_csv(self):
        text = df.describe().to_string()
        self.data_textedit.setPlainText(text)

    def clear_it(self):
        text = ""
        self.data_textedit.setPlainText(text)

    def numb_of_miss_value(self):
        text = df.isnull().sum().to_string()
        self.data_textedit.setPlainText(text)

    def print_all(self):
        for index, row in df.iterrows():
            row_text = str(row) + "\n\n"
            self.data_textedit.insertPlainText(row_text)

    def load_data(self):
        headers = df.columns.values.tolist()  # Nadpisy sloupců z prvního řádku

        # Nahrání nadpisů sloupců do prvního QComboBoxu
        for header in headers:
            self.combo_box1.addItem(header)

        # Nahrání nadpisů sloupců do druhého QComboBoxu
        for header in headers:
            self.combo_box2.addItem(header)

        # Nahrání nadpisů sloupců do prvního QComboBoxu
        for header in headers:
            self.combo_box3.addItem(header)

        # Nahrání nadpisů sloupců do druhého QComboBoxu
        for header in headers:
            self.combo_box4.addItem(header)

    def vztah(self):
        selected_option1 = self.combo_box1.currentText()
        selected_option2 = self.combo_box2.currentText()
        plt.clf()
        sns.scatterplot(x=f"{selected_option1}", y=f"{selected_option2}", data=df)
        plt.title(f"Vztah mezi {selected_option1} a {selected_option2}")
        plt.savefig('graph.png')

        # Načtení obrázku
        image = cv2.imread('graph.png')
        # Vytvoření vyskakovacího okna a zobrazení obrázku
        cv2.imshow(f"Vztah mezi {selected_option1} a {selected_option2}", image)

        # Počkejte na stisk klávesy pro ukončení programu
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # Klávesa "Esc"
                break

        # Zavření vyskakovacího okna
        cv2.destroyAllWindows()

    def Save(self):
        timestamp = time.strftime("%Y%m%d%H%M%S")  # Časové razítko ve formátu YYYYMMDDHHMMSS
        filename = f"graph_{timestamp}.png"  # Unikátní název souboru
        plt.savefig(filename)

    def vztah_s_predikci(self):
        selected_option1 = self.combo_box3.currentText()
        selected_option2 = self.combo_box4.currentText()
        plt.clf()
        X = np.array(df[f"{selected_option1}"]).reshape(-1, 1)
        Y = np.array(df[f"{selected_option2}"])

        model = LinearRegression()
        print(model.fit(X, Y))

        # Predikce výstupů na základě vstupních dat
        y_pred = model.predict(X)

        # Vykreslení originálních hodnot a predikovaných hodnot
        plt.scatter(X, Y, color='blue')
        plt.plot(X, y_pred, color='red', linewidth=2)

        # Nastavení popisků os a titulku grafu
        plt.xlabel(f"{selected_option1}")
        plt.ylabel(f"{selected_option2}")
        plt.title(f"Vztah mezi {selected_option1} a {selected_option2} \ns predikovanými hodnotami")

        plt.savefig('graph_pred.png')

        # Načtení obrázku
        image = cv2.imread('graph_pred.png')
        # Vytvoření vyskakovacího okna a zobrazení obrázku
        cv2.imshow(f"Vztah mezi {selected_option1} a {selected_option2}", image)

        # Počkejte na stisk klávesy pro ukončení programu
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # Klávesa "Esc"
                break

        # Zavření vyskakovacího okna
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    analyze = AnalyzeApp()
    analyze.show()
    sys.exit(app.exec())
