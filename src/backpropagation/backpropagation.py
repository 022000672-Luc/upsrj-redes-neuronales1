# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: backpropagation.py
# Descripción: Definición del algoritmo de retropropagación (backpropagation) de una red neuronal.
# ============================================================
import sys, os, random
import numpy as np
from perceptron import InputData, Perceptron
#############################################################################################################################
# Algoritmo de retropropagación (backpropagation) en una red neuronal lineal                                                #
#                                                                                                                           #
# Objetivo: Implementar el cálculo de los gradientes y la actualización de pesos en función del error de salida.            #
#                                                                                                                           #
# Consideraciones:                                                                                                          #
# - La red ya ha realizado la propagación hacia adelante y ha generado un "network_output".                                 #
# - Se conoce el valor esperado ("expected_output") para esa entrada.                                                       #
# - Se deben calcular los errores hacia atrás desde la neurona de salida hasta las capas ocultas.                           #
#                                                                                                                           #
# El flujo esperado del algoritmo es:                                                                                       #
#                                                                                                                           #
# 1. Calcular el error en la neurona de salida:                                                                             #
#    error_salida = expected_output - network_output                                                                        #
#                                                                                                                           #
# 2. Calcular el gradiente de la neurona de salida:                                                                         #
#    delta_salida = error_salida * derivada_de_la_función_de_activación                                                     #
#                                                                                                                           #
# 3. Para cada capa oculta (en orden inverso):                                                                              #
#    3.1. Calcular el error de cada neurona como la suma ponderada de los deltas de la capa siguiente.                      #
#    3.2. Calcular el delta de cada neurona usando la derivada de su función de activación.                                 #
#                                                                                                                           #
# 4. Actualizar los pesos de cada conexión:                                                                                 #
#    nuevo_peso = peso_actual + tasa_de_aprendizaje * delta * entrada_correspondiente                                       #
#                                                                                                                           #
# 5. Repetir el proceso para cada muestra del conjunto de entrenamiento.                                                    #
#                                                                                                                           #
# NOTA: Este algoritmo permite que la red aprenda ajustando sus pesos para minimizar el error de salida.                    #
#       Se recomienda modularizar el código y documentar cada paso con ejemplos y analogías.                                #
#############################################################################################################################

# Ejercicio
# TODO: Define una funcion "backpropagation_network" que implemente una red
#       neuronal de propagación lineal con retropropagación, considera las clases "InputData" y "Perceptron"
#       definidas en el paquete "perpetron".
# 
#       Los parámetros de la función deben ser los siguientes:
#       - inputs (list): entradas que tendrá la red
#       - perceptrons (int): número de neuronas que tendrá cada capa de la red
#       - layers (int): número de capas que tendrá la red
#
#       Las salidas de la función deben ser las siguientes: 
#       - network_output (float): cálculo de "a" de la capa de salida.
#
# Ejemplo de uso:
#   inputs = np.array([0.5, 0.8, 0.2])
#   perceptrons = 4
#   layers = 3
#   output = backpropagation_network(inputs, perceptrons, layers)
def backpropagation_network(inputs:np.ndarray, perceptrons:int, layers:int) -> float:
    # Mensaje para identificar que entramos exitosamente a la función
    print("corriendo red de retropropagación con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(inputs=inputs, perceptrons=perceptrons, layers=layers))
    # Escribe tu código aquí
    network_output = 0.0

    # Return de la función: cálculo de "a" de la capa de salida
    return float(network_output)