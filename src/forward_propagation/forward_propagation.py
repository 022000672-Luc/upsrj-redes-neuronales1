# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: forward_propagation.py
# Descripción: Definición del algoritmo de propagación linear (forward propagation)
#              de una red neuronal.
# ============================================================

import sys, os, random
import numpy as np
from perceptron.input_data import InputData
from perceptron.perceptron import Perceptron

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
# 2. Para cada capa oculta:                                                                                                 #
#    2.1. Si es la primera capa, cada neurona se conecta a cada entrada del arreglo inicial.                                #
#    2.2. Si no es la primera capa, cada neurona se conecta a cada salida de cada neurona de la capa anterior.              #
#    2.3. Las salidas de la capa se transforman en un arreglo de objetos InputData para alimentar la siguiente capa.        #
# 3. En la capa de salida hay una única neurona que recibe como entrada todas las salidas de la última capa oculta.         #
# 4. El valor de salida de la red ("network_output") corresponde al valor de activación ("a") de la neurona de salida.      #
#############################################################################################################################


# ===========================================================================================================================
# Función: forward_propagation_network
# Descripción:
#   Implementa una red neuronal de propagación lineal según los pasos descritos.
#   - Cada capa está compuesta por un número definido de perceptrones.
#   - Las salidas de una capa sirven como entradas para la siguiente.
#   - La última capa tiene una única neurona de salida.
# ===========================================================================================================================

def forward_propagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    """
    Implementación de una red neuronal de propagación hacia adelante (forward propagation)
    con estructura lineal.

    Parámetros:
        inputs (np.ndarray): Entradas iniciales hacia la red.
        perceptrons (int): Número de neuronas por capa.
        layers (int): Número de capas ocultas.

    Retorna:
        network_output (float): Valor de activación "a" de la neurona de salida.
    """

    # -----------------------------------------------------------------------------------------------------------
    # Paso 1: Mostrar configuración inicial
    # -----------------------------------------------------------------------------------------------------------
    print("corriendo red con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(
        inputs=inputs, perceptrons=perceptrons, layers=layers
    ))

    # La capa de entrada recibe los valores iniciales como lista
    current_x_values_for_next_layer = inputs.tolist()

    # -----------------------------------------------------------------------------------------------------------
    # Paso 2: Propagación a través de las capas ocultas
    # -----------------------------------------------------------------------------------------------------------
    for layer_idx in range(layers):
        layer_outputs_values = []  # almacena las salidas de la capa actual

        # Recorremos cada perceptrón de la capa
        for _ in range(perceptrons):
            perceptron_inputs_for_this_neuron = []

            # Paso 2.1 y 2.2: conectar cada neurona a las entradas correspondientes
            for x_val in current_x_values_for_next_layer:
                perceptron_inputs_for_this_neuron.append(InputData(x=float(x_val)))

            # Crear el perceptrón con un sesgo aleatorio pequeño
            p = Perceptron(inputs=perceptron_inputs_for_this_neuron, b=np.random.randn() * 0.01)

            # Ejecutar el perceptrón (calcula la activación)
            p.run()

            # Guardar salida "a" de la neurona actual
            layer_outputs_values.append(p.a)

        # Paso 2.3: las salidas de la capa actual se convierten en entradas para la siguiente capa
        current_x_values_for_next_layer = layer_outputs_values

    # -----------------------------------------------------------------------------------------------------------
    # Paso 3: Capa de salida (una sola neurona)
    # -----------------------------------------------------------------------------------------------------------
    final_perceptron_inputs = []
    for x_val in current_x_values_for_next_layer:
        final_perceptron_inputs.append(InputData(x=float(x_val)))

    # Crear la neurona de salida
    final_perceptron = Perceptron(inputs=final_perceptron_inputs, b=np.random.randn() * 0.01)
    final_perceptron.run()

    # -----------------------------------------------------------------------------------------------------------
    # Paso 4: Salida final de la red
    # -----------------------------------------------------------------------------------------------------------
    network_output = final_perceptron.a

    return float(network_output)
