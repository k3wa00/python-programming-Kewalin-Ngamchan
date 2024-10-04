# Grunduppgift

# Steg 1 
import pandas as pd 

# Läser in datapunkterna från txt-fil med pandas och separarerar med komma
# Skippar första raden på grund av felaktig uppdelning av kolumnen "label"
data = pd.read_csv("/Users/ngamchan/Downloads/datapoints.txt" , delimiter =",", skiprows=1)

# Skapar lista med nya labels 
labels = ["width", "height", "label"] 

# Tilldelar kolumnnamn till data
data.columns = labels

# Kontroll för att se att datan skrivs ut rätt
print(data.head())

#källa: w3schools 
#länk: https://www.w3schools.com/python/pandas/pandas_csv.asp
#källa: https://www.geeksforgeeks.org/remove-spaces-from-column-names-in-pandas/


# Steg 2 
# Importerar matplotlib och förkortar det som "plt" för enkelhetens skull
import matplotlib.pyplot as plt 


# Separerar datan och skapar två separata dataset
# Använder mig av indexering och tar rader från orginal datan (labels) som uppfyller kravet
pichu = data[data["label"] == 0]
pikachu = data[data["label"] == 1]

# Skapar scatter plot för att visualisera data, skillnaden mellan Pichu och Pikachu
plt.scatter(pichu["width"], pichu["height"], color="blue", label="Pichu")
plt.scatter(pikachu["width"], pikachu["height"], color="red", label="Pikachu")

# Ger titlar till axlarna 
plt.ylabel("height")
plt.xlabel("width")
plt.legend()



#källa: https://www.geeksforgeeks.org/data-visualization-with-python/


# Steg 3 
import pandas as pd

# Läser in testdata 
testdata = pd.read_csv("/Users/ngamchan/Downloads/testpoints.txt", delimiter=",")
print(testdata.columns)

# Beräknar medelvärden för Pikachu och Pichu för att kunna beräkna avstånd  

pikachu_m = {"width": [25, 24.2, 22], 
              "height": [32, 31.5, 34]}

pichu_m = {"width": [20.5], 
            "height": [34]}

print("Pikachu means:", pikachu_m)
print("Pichi means:", pichu_m)


# Steg 4 
import math

# Euclidean distance 
# Skapar en funktion för att beräkna avstånd mellan Pikachu och Pichu baserat på medelvärden
def distance(pikachu_m, pichu_m):
    # Initiera totalavståndet till 0 
    # Viktigt för annars får man inte rätt summa av alla avstånd
    total_distance = 0.0

    # Iterera över varje Pikachu-punkt
    for i in range(len(pikachu_m["width"])):


        # Beräkna skillnaden i bredd mellan Pichu och aktuell Pikachu-punkt

        width_diff = (pichu_m['width'][0] - pikachu_m['width'][i]) ** 2

        # Beräkna skillnaden i höjd mellan Pichu och aktuell Pikachu-punkt

        height_diff = (pichu_m['height'][0] - pikachu_m['height'][i]) ** 2


        # Lägg till det euklidiska avståndet för den aktuella punkten till totalavståndet

        total_distance += math.sqrt(width_diff + height_diff)

# Returnera det totala avståndet     
    return total_distance

print("Avståndet mellan Pikachu och Pichu:", distance(pikachu_m, pichu_m))

# Steg 5 
# Tränar modellen med logisk regression för att klassificera datapunkterna tillhör Pikachu eller Pichu

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np

# Definerar variablerna med deras värden
# 'x' innehåller egenskaperna som används för att förutsäga etiketten

x = data[["width" , "height"]] 

# 'y' innehåller målet (etiketten) som ska förutsägas
y = data["label"]

# Dela upp data i tränings- och testset (80% träning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Skapa en logistisk regressionsmodell
clf = LogisticRegression(random_state=0)

# Träna modellen med träningsdata
clf.fit(X_train, y_train)

# Gör förutsägelser på testdata
y_pred = clf.predict(X_test)

# Beräknar noggrannheten för modellen
accuracy = accuracy_score(y_test, y_pred)

# Avrundar decimalen och skriv ut noggrannheten

print(f"Modellens noggrannhet: {accuracy:.2f}")



#https://www.geeksforgeeks.org/understanding-logistic-regression/#code-implementation-for-logistic-regression 

# uppgift 1: Användarinput

# Använder en while-loop för att fortsätta be om användarens input
# tills giltiga värden (större än 0) matas in, vilket förhindrar att programmet kraschar.

while True:
    try:
        w = float(input("Ange en bredd:"))
        h = float(input("Ange en höjd: "))

        
        # Kontrollera att både bredd och höjd är större än 0 (större än 0)
        # För att säkerställa giltig input 

        if w > 0 and h > 0:
            break     #  Avbryt loopen om giltiga värden anges
        else:
            print("Värderna måste vara större än 0. Försök igen!") 
    except ValueError:
        print("Fel inmatning! Skriv endast siffror.")


plt.show()
#källa https://stackoverflow.com/questions/38040898/how-to-get-only-numbers-as-a-response-in-python

# Skapar en numpy-array för att använda med modellen
user_input = np.array([[w,h]])

# Gör en förutsägelse baserat på den angivna punkten
pred_input_result = clf.predict(user_input)

# Avgör tillhörande klass baserat på modellen
if pred_input_result[0] == 0:
    print("Den nya punkten tillhör Pichu!")
else:
    print("Den nya punkten tillhör Pikachu!")


# Uppgift 2: Klassicifering med KNN-algoritm 

# Använder KNN-algoritmen för att klassificera de 10 närmaste grannarna
# 'n_neighbors' sätts till 10 för att hitta de 10 närmaste punkterna
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)   # Initiera KNN-modellen
model.fit(X_train, y_train)    # Träna modellen med träningsdata

# Definiera en ny punkt för att göra förutsägelser
new_point = np.array([[0,11]])

# Gör förutsägelser på den nya punkten
new_pred_input= model.predict(new_point)

# Beräkna noggrannheten på modellen med testdata
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Avgör tillhörande klass för den nya punkten
if new_pred_input == 0:
    print("Den nya punkten tillhör Pichu!")
else:
    print("Den nya punkten tillhör Pikachu!")

#källa: https://www.w3schools.com/python/python_ml_knn.asp
#källa: https://apmonitor.com/pds/index.php/Main/KNearestNeighbors 



