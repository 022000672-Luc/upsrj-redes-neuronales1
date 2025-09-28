# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: forward_propagation.py
# Descripción: Definición del algoritmo de propagación linear (forward propagation) de una red neuronal.
# ============================================================
import sys, os, random
import numpy as np
from perceptron import InputData, Perceptron
#############################################################################################################################
# Algoritmo de propagación hacia adelante (forward propagation) en una red neuronal lineal                                  #
#                                                                                                                           #
# En una red de propagación lineal se definen:                                                                              #
# - n entradas hacia la red                                                                                                 #
# - n neuronas por capa                                                                                                     #
# - n capas ocultas                                                                                                         #
#                                                                                                                           #
# El flujo del algoritmo se describe así:                                                                                   #
#                                                                                                                           #
# 1. La capa de entrada recibe un arreglo de números flotantes, que se convierte en un arreglo de objetos InputData.        #
#                                                                                                                           #
# 2. Para cada capa oculta:                                                                                                 #
#    2.1. Si es la primera capa, cada neurona se conecta a cada entrada del arreglo inicial.                                #
#    2.2. Si no es la primera capa, cada neurona se conecta a cada salida de cada neurona de la capa anterior.              #
#    2.3. Las salidas de la capa se transforman en un arreglo de objetos InputData para alimentar la siguiente capa.        #
#                                                                                                                           #
# 3. En la capa de salida hay una única neurona que recibe como entrada todas las salidas de la última capa oculta.         #
#                                                                                                                           #
# 4. El valor de salida de la red ("network_output") corresponde al valor de activación ("a") de la neurona de salida.      #
#############################################################################################################################

# Ejercicio
# TODO: Define una funcion "forward_propagation_network" que implemente una red
#       neuronal de propagación lineal, considera las clases "InputData" y "Perceptron"
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
#   output = forward_propagation_network(inputs, perceptrons, layers)
def forward_propagation_network(inputs:np.ndarray, perceptrons:int, layers:int) -> float:
    # Mensaje para identificar que entramos exitosamente a la función
    print("corriendo red con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(inputs=inputs, perceptrons=perceptrons, layers=layers))
    # Escribe tu código aquí
    network_output = 0.0

    # Return de la función: cálculo de "a" de la capa de salida
    return float(network_output)