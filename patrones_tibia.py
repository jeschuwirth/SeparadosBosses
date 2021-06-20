import pickle
import matplotlib.pyplot as plt
import numpy as np
from lbp import lbp_features



def comparar(img_path):
    
    BOSSESINDB = {
    0:"DC",
    1:"Drume",
    2:"Faceless",
    3:"GD",
    4:"GT",
    5:"Last DC",
    6:"Leao",
    7:"Scarlett",
    8:"Wz4",
    9:"Wz5",
    10:"Wz6"
    }

    img = plt.imread(img_path)
    Xlbp = np.zeros((1, 3 * 10))
    Xlbp[0, 0:10] = lbp_features(img[:,:,0], hdiv=1, vdiv=1, mapping='uniform')
    Xlbp[0, 10:20] = lbp_features(img[:,:,1], hdiv=1, vdiv=1, mapping='uniform')
    Xlbp[0, 20:30] = lbp_features(img[:,:,2], hdiv=1, vdiv=1, mapping='uniform')

    lda = pickle.load(open('trained LDA.sav', 'rb'))
    result = lda.predict(Xlbp)

    #Nota, esto es sólo lda y extracción de caracerísticas
    #No hay limpieza ni selección
    return BOSSESINDB[result[0]]

path = "C:/Users/jschu/Desktop/New folder/Scarlett"
img = "2021-06-18_075438348_Captain Neca_BossDefeated.png"
print(comparar(f"{path}/{img}"))

