import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from classifier import datapoint_classification  # Importera klassificeringsfunktionen


# Läs in data
filnamn = "unlabelled_data.csv"
data = pd.read_csv(filnamn, sep=",", header=None)
data.columns = ["X", "Y"]

print(data.head())

# Skapa scatter plot
plt.scatter(data["X"], data["Y"], color='blue', label='Data Points')

# Välj två specifika datapunkter (andra raden och första raden)
P1 = (2.090362, 2.562490)  # Andra raden
P2 = (-1.885908, -1.997408)  # Första raden

# Beräkna lutning (slope) baserat på dessa två punkter
slope = (P2[1] - P1[1]) / (P2[0] - P1[0])

# Beräkna intercept (m) baserat på en av punkterna
intercept = P1[1] - slope * P1[0]

# Plotta linjen baserat på dessa värden
line = slope * data["X"] + intercept
plt.plot(data["X"], line, color="red", label=f"Linje: y = {slope:.2f}x + {intercept:.2f}")

# Titel och axlar
plt.title("Scatter Plot med Linje")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Linjens ekvation: y = {slope:.2f}x + {intercept:.2f}")

# Implementera en funktion som räknar ut om en punkt ligger ovanför eller nedanför linjen 
# classifier.py
def datapoint_classification(points, slope, intercept):
    classifications = []
    for x_i, y_i in points:
        # Beräkna linjens y-värde 
        y_line = slope * x_i + intercept 

        # Jämför y-värde 
        if y_i > y_line:
            classifications.append(1)  # Ovanför linjen
        else:
            classifications.append(0)  # Nedanför eller på linjen
    return classifications

# Kalla på funktionen 
data['Class'] = datapoint_classification(data[["X", "Y"]].values, slope, intercept)

# Skriv till en fil labelled_data.csv
data.to_csv("labelled_data.csv", index=False)

# Plotta alla punkterna med deras klass och linjen
plt.scatter(data["X"], data["Y"], c=data['Class'], cmap='bwr', label='Data Points')
plt.plot(data["X"], line, color="red", label=f"Linje: y = {slope:.2f}x + {intercept:.2f}")
plt.title("Klassificerade Data Punkter")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

