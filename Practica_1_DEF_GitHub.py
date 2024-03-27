#José Francisco Andrade Soto
#Daniel Ríos Meizoso

#Instrucciones
#Crear una carpeta de nombre 'Graficas'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def leer_conjunto_datos(nombre_archivo,columnas_leer):
    df = pd.read_csv(nombre_archivo, usecols=columnas_leer)
    return df

def dimensiones(df):
    filas,columnas= df.shape
    return filas,columnas

def contador_de_valores_nulos(df):
    return df.isnull().sum()

def suprimir_filas_valores_nulos(df):
    #Si fuese axis=1 se eliminarían las columnas con valores nulos.
    df_v1=df.dropna(axis=0)
    return df_v1

def contador_muestras_por_clase(df_v1,clase):
    return df_v1[clase].value_counts()

def definir_entradas(df_v1,clase):
    #Las características son todas las columnas menos la última, debido a que es la clase.
    caracteristicas_dfv1 = df_v1[df_v1.columns[0:df_v1.shape[1]-1]]
    X = np.asarray(caracteristicas_dfv1)
    #Clase
    y = np.asarray(df_v1[clase])
    return X,y 

def normalizar(X):
    scaler=preprocessing.StandardScaler()
    X_norm=scaler.fit_transform(X)
    return X_norm

def P_C_A(max_componentes,X):
    var_comp=[]
    for i in range(max_componentes+1):
        pca=PCA(i)
        pca.fit(X)
        varianza_acumulada=pca.explained_variance_ratio_.sum()
        var_comp.append(varianza_acumulada)
        print(f'Para {i+1} componentes posee de varianza acumulada: {var_comp[i]}') 
    return var_comp

def eleccion_componente(var_comp):
    #El bucle se interrumpe cuando un valor de la lista es superior a 0.95 de varianza. 
    #Esta función devuelve el índice de la lista que contiene el primer valor superior a 0.95.
    indice_superior = None
    for indice, valor in enumerate(var_comp):
        if valor > 0.95:
            indice_superior = indice
            break

    return indice_superior

def reduccion_dimensionalidad(X,componentes):
    pca = PCA(n_components=componentes)  
    X_reducido = pca.fit_transform(X)
    return X_reducido 


def K_N_N(X,y,k,nombre):

    lista_n_vecinos = range(1, k)
    # Inicializar listas para almacenar las medias y desviaciones estándar de la exactitud
    lista_medias = []
    lista_desviaciones = []

    for n_vecinos in lista_n_vecinos:
        accuracies = []
        tiempo_inicial = time.time()  # Iniciar el contador de tiempo
        for random_state in range(1, 101):  # 100 random states consecutivos del 1 al 100

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
            
            # Inicializar y entrenar el modelo de clasificación (k-NN)
            model = KNeighborsClassifier(n_neighbors=n_vecinos)
            model.fit(X_train, y_train)
            
            # Hacer predicciones en el conjunto de prueba
            y_pred = model.predict(X_test)
            
            # Evaluar el rendimiento del modelo
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        tiempo_final = time.time()  # Finalizar el contador de tiempo
        tiempo_transcurrido = tiempo_final- tiempo_inicial  # Calcular el tiempo transcurrido
        print(f"Tiempo de entrenamiento para {n_vecinos} vecinos: {tiempo_transcurrido:.4f} segundos")
        
        # Calcular la media de la exactitud para este valor de n_vecinos
        media_exactitud = np.mean(accuracies)
        lista_medias.append(media_exactitud)
        print(f'Media de exactitud para {n_vecinos} vecinos: {media_exactitud} ')
        
        # Calcular la desviación estándar de la exactitud para este valor de n_vecinos
        desviacion_exactitud = np.std(accuracies)
        lista_desviaciones.append(desviacion_exactitud)

    # Graficar las medias de la exactitud y las desviaciones estándar para cada valor de n_neighbors
    grafica_media_y_desviacion_KNN(lista_n_vecinos, lista_medias,lista_desviaciones,nombre)
   
    confusion_matrix(y_test, y_pred, labels=[1,2,3])
    

def S_V_M(X,y,C_values,met,nombre):
    

    # Inicializar listas para almacenar las medias y desviaciones estándar de la exactitud
    mean_accuracies = []
    std_accuracies = []

    for C in C_values:
        accuracies = []
        start_time = time.time()  # Iniciar el contador de tiempo
        for random_state in range(1, 100):  # 100 random states consecutivos del 1 al 100
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
            
            # Inicializar y entrenar el modelo SVM
            model = SVC(C=C, kernel=met, gamma='scale', random_state=random_state)
            model.fit(X_train, y_train)
            
            # Hacer predicciones en el conjunto de prueba
            y_pred = model.predict(X_test)
            
            # Evaluar el rendimiento del modelo
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        end_time = time.time()  # Finalizar el contador de tiempo
        elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido
        print(f"Tiempo de entrenamiento para C={C}: {elapsed_time:.4f} segundos")

        # Calcular la media de la exactitud para este valor de C
        mean_accuracy = np.mean(accuracies)
        mean_accuracies.append(mean_accuracy)
        print(f'Media de exactitud para C={C}: { mean_accuracy} ')
        
        # Calcular la desviación estándar de la exactitud para este valor de C
        std_accuracy = np.std(accuracies)
        std_accuracies.append(std_accuracy)

    # Graficar las medias de la exactitud y las desviaciones estándar para cada valor de C
    grafica_media_y_desviacion_SVM(C_values,mean_accuracies,std_accuracies,nombre)
    #Graficar la matriz de confusión
    matriz_confusion(y_test, y_pred,nombre)
    return mean_accuracies,std_accuracies


def entrenamiento(X,y,random):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=random)

    return X_train, X_test, y_train, y_test

def matriz_confusion(y_test, y_pred,nombre):
    cm_SVM = confusion_matrix(y_test, y_pred, labels=[1,2,3])
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM, display_labels=['Coruña Dique(1)','Cebreiro(2)','Sant San Lazaro(3)'])
    disp_SVM.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión k-NN")
    plt.savefig('Graficas/'+'Matriz_Confusión_'+str(nombre)+'.jpg')
    plt.close()

def grafica_media_y_desviacion_KNN(lista_n_vecinos, lista_medias,lista_desviaciones,nombre):
    plt.errorbar(lista_n_vecinos, lista_medias, yerr=lista_desviaciones, fmt='-o', ecolor='orange')
    plt.title('Media y desviación estándar de la exactitud en función  k-vecinos')
    plt.xlabel('k-vecinos')
    plt.ylabel('Exactitud media')
    plt.xticks(lista_n_vecinos)
    plt.grid(True)
    plt.savefig('Graficas/'+str(nombre)+'.jpg')
    plt.close()

def grafica_media_y_desviacion_SVM(C_values,mean_accuracies,std_accuracies,nombre):
    plt.errorbar(C_values, mean_accuracies, yerr=std_accuracies, fmt='-o', ecolor='orange')
    plt.title('Media y desviación típica de la exactitud para distintos C valores')
    plt.xlabel('C valores')
    plt.ylabel('Exactitud media')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('Graficas/'+str(nombre)+'.jpg')
    plt.close()


def grafica_comparativa(C_values,lista_media_rbf,lista_desviacion_rbf,lista_media_poly,lista_desviacion_poly, lista_media_sigmoid,lista_desviacion_sigmoid):
    plt.errorbar(C_values, lista_media_rbf, yerr=lista_desviacion_rbf, fmt='-o', label='[SVM] rbf')
    plt.errorbar(C_values, lista_media_poly, yerr=lista_desviacion_poly, fmt='-o', label='[SVM] poly')
    plt.errorbar(C_values, lista_media_sigmoid, yerr=lista_desviacion_sigmoid, fmt='-o', label='[SVM] sigmoid')

    plt.title('Media y desviación típica en función de C valores para distintos tipos de SVM')
    plt.xlabel('C valores')
    plt.ylabel('Exactitud Media')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('Graficas/Comparacion_tipos_SVM.jpg')
    plt.close()

def main():
    #Una vez observado el archivo csv y establecido las características de interés, 
    #se procede a la lectura del mismo.
    #Se le pasará a la función 'leer_conjunto_datos' el nombre del archivo csv y las columnas de dicho archivo
    #que poseen características a analizar.Esta función devolverá un DataFrame.
    #Interesa leer todas las columnas a excepción de:
    #[1] Nombre del mes [2] Instante de la lectura de los datos [13] Nombre de la estación
    nombre_archivo='meteosinseparar.csv'
    columnas_leer=[0,3,4,5,6,7,8,9,10,11,12]
    df=leer_conjunto_datos(nombre_archivo,columnas_leer)

    #Dimensiones del DataFrame de partida.
    print('\nConjunto de datos de partida:\n')
    print(df)
    filas,columnas=dimensiones(df)
    print(f'\nEl conjunto de datos de partida posee: {filas} filas y {columnas} columnas\n')

    #Se procede a contabilizar los valores nulos en cada columna para la posterior eliminación de la fila 
    #que los posee. Se obtiene un DataFrame modificado 'df_v1'
    n_valores_nulos=contador_de_valores_nulos(df)
    print('\nSe muestra el número de valores nulos por columnas:\n')
    print(n_valores_nulos)
    df_v1=suprimir_filas_valores_nulos(df)
    filas_v1,columnas_v1=dimensiones(df_v1)
    print(f'\nEl conjunto de datos suprimiendo las filas con valores nulos posee: {filas_v1} filas y {columnas_v1} columnas\n')

    #Se realiza un conteo del número de muestras por clase. Este procedimiento se hace con el fin de establecer
    #la necesidad o no de realizar un Balanceo. La columna que contiene la clase es cuya cabecera posee el nombre 
    # de Estación
    clase='Estación'
    m_valores_clase=contador_muestras_por_clase(df_v1,clase)
    print('Se muestra el número de valores por clase: \n')
    print(m_valores_clase)

    #Se definen las características y de las clases.
    X,y=definir_entradas(df_v1,clase)
    filas_X,columnas_X=dimensiones(X)
    print('\nCaracterísticas NO normalizadas:\n')
    print(X)
    print('\nClase:\n')
    print(y)
    
    #Normalización de las características
    X_norm=normalizar(X)
    print('\nCaracterísticas normalizadas:\n')
    print(X_norm)
    
    #INICIO DE LAS TÉCNICAS DE APRENDIZAJE
    #1)SIN PCA Y SIN NORMALIZAR
    print('\n SIN PCA Y SIN NORMALIZAR\n')

    #1A)k-NN SIN PCA y SIN NORMALIZACIÓN 
    print('\n Técnica: k Nearest Neighbors [k-NN] sin PCA y sin normalización\n')
    k=21
    K_N_N(X,y,k,'k-NN_Sin_Normalizar_Sin_PCA')

    #1B)SVM rbf SIN PCA y SIN NORMALIZACIÓN 
    C_values = [0.1, 1, 10, 100, 1000]
    print('\n Técnica: Support Vector Machine [SVM] rbf sin PCA y sin normalizado \n')
    S_V_M(X,y,C_values,'rbf','SVM_rbf_Sin_Normalizar_Sin_PCA')

    #2)CON PCA Y SIN NORMALIZAR
    #PCA (Análisis de Componentes Principales) y métodos SVM y KNN sin normalizar.
    print('\nCON PCA Y SIN NORMALIZACIÓN\n')

    #Se procede a calcular la varianza para distintas cantidades de componentes.La función 'P_C_A'
    #devuelve una lista que contiene la varianza para distintas cantidades de componentes.

    lista_var_PCA=P_C_A(columnas_X,X)

    #Para saber el número de componentes a necesitar, se estableció que el número de componentes será igual
    #al índice+1 del primer valor de la lista anteriormente obtenida cuyo valor sea superior a 0.95.

    indice_lista_var_PCA=eleccion_componente(lista_var_PCA)
    n_componente_PCA=indice_lista_var_PCA+1
    print(f'\nSe utilizaran {n_componente_PCA} componentes\n')

    #Se reduce la dimensionalidad según lo establecido en PCA.

    X_reducido=reduccion_dimensionalidad(X,n_componente_PCA)
    print(f'\nCaracterísticas NO normalizadas reducidas a {n_componente_PCA} componentes:\n')
    print(X_reducido)
    print('\n')

    #2A)KNN CON PCA Y SIN NORMALIZACIÓN
    print(f'\n Técnica: k Nearest Neighbors [k-NN] con {n_componente_PCA} PCA y sin normalizado\n ')
    k=21
    K_N_N(X_reducido,y,k,'k-NN_Sin_Normalizar_con_'+str(n_componente_PCA)+'-PCA')

    #2B) SVM rbf N SIN NORMALIZACIÓN
    C_values = [0.1, 1, 10, 100, 1000]
    print(f'\n Técnica: Support Vector Machine [SVM] rbf con {n_componente_PCA} PCA y sin normalizado \n')
    S_V_M(X_reducido,y,C_values,'rbf','SVM_rbf_Sin_Normalizar_con_'+str(n_componente_PCA)+'-PCA')

    #3) CON PCA (Análisis de Componentes Principales) y NORMALIZANDO.
    
    print('\n CON PCA Y NORMALIZACIÓN\n')

    #Se procede a calcular la varianza para distintas cantidades de componentes.La función 'P_C_A'
    #devuelve una lista que contiene la varianza para distintas cantidades de componentes.

    lista_var_PCA_norm=P_C_A(columnas_X,X_norm)

    #Para saber el número de componentes a necesitar, se estableció que el número de componentes será igual
    #al índice+1 del primer valor de la lista anteriormente obtenida cuyo valor sea superior a 0.95.

    indice_lista_var_PCA_norm=eleccion_componente(lista_var_PCA_norm)
    n_componente_PCA_norm=indice_lista_var_PCA_norm+1
    print(f'\nSe utilizaran {n_componente_PCA_norm} componentes\n')

    #Se reduce la dimensionalidad según lo establecido en PCA.
    X_reducido_norm=reduccion_dimensionalidad(X_norm,n_componente_PCA_norm)
    print(f'\nCaracterísticas normalizadas reducidas a {n_componente_PCA_norm} componentes:\n')
    print(X_reducido_norm)
    print('\n')

    #3A)KNN con PCA Y NORMALIZADO
    print(f'\n Técnica: k Nearest Neighbors [k-NN] con {n_componente_PCA_norm} PCA y normalización\n ')
    k=21
    K_N_N(X_reducido_norm,y,k,'k-NN_Normalizado_con_'+str(n_componente_PCA_norm)+'-PCA')

    #3B)SVM rbf con PCA Y NORMALIZADO
    print(f'\n Técnica: Support Vector Machine [SVM] rbf con {n_componente_PCA_norm} PCA y normalización \n')
    lista_media_rbf,lista_desviacion_rbf=S_V_M(X_reducido_norm,y,C_values,'rbf','SVM_rbf_Normalizado_con_'+str(n_componente_PCA_norm)+'-PCA')

    #3C)SVM poly con PCA Y NORMALIZADO
    print(f'\n Técnica: Support Vector Machine [SVM] poly con {n_componente_PCA_norm} PCA y normalización \n')
    lista_media_poly,lista_desviacion_poly=S_V_M(X_reducido_norm,y,C_values,'poly','SVM_poly_Normalizado_con_'+str(n_componente_PCA_norm)+'-PCA')

    #3D)SVM sigmoid con PCA Y NORMALIZADO
    print(f'\n Técnica: Support Vector Machine [SVM] sigmoid con {n_componente_PCA_norm} PCA y normalización\n')
    lista_media_sigmoid,lista_desviacion_sigmoid=S_V_M(X_reducido_norm,y,C_values,'sigmoid','SVM_sigmoid_Normalizado_con_'+str(n_componente_PCA_norm)+'-PCA')

    grafica_comparativa(C_values,lista_media_rbf,lista_desviacion_rbf,lista_media_poly,lista_desviacion_poly, lista_media_sigmoid,lista_desviacion_sigmoid)
    

if __name__=='__main__':
    main()