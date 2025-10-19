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
from perceptron.input_data import InputData
from perceptron.perceptron import Perceptron


def forward_propagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    print("corriendo red con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(
        inputs=inputs, perceptrons=perceptrons, layers=layers
    ))

    current_x_values_for_next_layer = inputs.tolist()

    for layer_idx in range(layers):
        layer_outputs_values = []

        for _ in range(perceptrons):
            perceptron_inputs_for_this_neuron = []
            for x_val in current_x_values_for_next_layer:
                perceptron_inputs_for_this_neuron.append(InputData(x=float(x_val)))

            p = Perceptron(inputs=perceptron_inputs_for_this_neuron, b=np.random.randn() * 0.01)
            p.run()
            layer_outputs_values.append(p.a)

        current_x_values_for_next_layer = layer_outputs_values

    final_perceptron_inputs = []
    for x_val in current_x_values_for_next_layer:
        final_perceptron_inputs.append(InputData(x=float(x_val)))

    final_perceptron = Perceptron(inputs=final_perceptron_inputs, b=np.random.randn() * 0.01)
    final_perceptron.run()

    network_output = final_perceptron.a

    return float(network_output)
