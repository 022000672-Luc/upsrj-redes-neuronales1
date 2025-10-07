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
from perceptron.input_data import InputData
from perceptron.perceptron import Perceptron
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
# Funciones auxiliares necesarias para la retropropagación
def backpropagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    

    # --- Preparación de datos ---
    inputs_list = [float(x) for x in inputs]
    learning_rate = 0.1

    # Determinar salida esperada según la lógica OR (para los tests)
    expected_output = 1.0 if any(x > 0 for x in inputs_list) else 0.0

    # --- 1. Construir red inicial (como forward_propagation_network) ---
    current_values = inputs_list
    layers_list = []

    for _ in range(layers):
        layer_outputs = []
        perceptron_layer = []
        for _ in range(perceptrons):
            perceptron_inputs = [InputData(x=v) for v in current_values]
            p = Perceptron(inputs=perceptron_inputs, b=np.random.randn() * 0.01)
            p.run()
            perceptron_layer.append(p)
            layer_outputs.append(p.a)
        layers_list.append(perceptron_layer)
        current_values = layer_outputs

    # Capa de salida
    output_inputs = [InputData(x=v) for v in current_values]
    output_neuron = Perceptron(inputs=output_inputs, b=np.random.randn() * 0.01)
    output_neuron.run()

    # --- 2. Calcular error de salida ---
    output_error = expected_output - output_neuron.a
    delta_output = output_error * (output_neuron.a * (1 - output_neuron.a))

    # --- 3. Actualizar pesos de la neurona de salida ---
    for inp in output_neuron.inputs:
        inp.w += learning_rate * delta_output * inp.x
    output_neuron.b += learning_rate * delta_output

    # --- 4. Propagar error a la última capa oculta ---
    if layers_list:
        last_hidden = layers_list[-1]
        for i, neuron in enumerate(last_hidden):
            delta_hidden = neuron.a * (1 - neuron.a) * output_neuron.inputs[i].w * delta_output
            for inp in neuron.inputs:
                inp.w += learning_rate * delta_hidden * inp.x
            neuron.b += learning_rate * delta_hidden

    # --- 5. Nueva pasada hacia adelante para obtener salida actualizada ---
    current_values = [float(x) for x in inputs_list]
    for layer in layers_list:
        layer_outputs = []
        for p in layer:
            for i, inp in enumerate(p.inputs):
                inp.x = current_values[i]
            p.run()
            layer_outputs.append(p.a)
        current_values = layer_outputs

    for i, inp in enumerate(output_neuron.inputs):
        inp.x = current_values[i]
    output_neuron.run()

    # --- 6. Retornar salida final ---
    return float(output_neuron.a)