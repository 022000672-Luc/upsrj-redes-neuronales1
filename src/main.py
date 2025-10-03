# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: main.py
# Descripción: Script principal
# ============================================================
import sys, os, random
import numpy as np
from forward_propagation.forward_propagation import forward_propagation_network
from backpropagation.backpropagation import backpropagation_network

def main():
    
    # Status: OK
    status = os.EX_OK

    # Entradas de la red neuronal: 
    # - funciones AND & OR 
    # - 100 perceptrones por capa 
    # - 100 capas
    funcion_and = np.array([0, 0, 0, 1])
    funcion_or = np.array([0, 1, 1, 1])
    perceptrones = 100
    capas = 100
    
    try:
        # Test case: 
        # - Entrada: backpropagation_network(inputs=funcion_and, perceptrons=perceptrones, layers=capas)
        # - Salida esperada: float 0 ~ 1
        # - Propósito: Ejecución de la red neuronal aplicada en la función "AND"
        red_neuronal_and = forward_propagation_network(inputs=funcion_and, perceptrons=perceptrones, layers=capas)
        print(f"Red neuronal 'forward' para AND generada. Resultado: {red_neuronal_and}\n")
        red_neuronal_and = backpropagation_network(inputs=funcion_and, perceptrons=perceptrones, layers=capas)
        print(f"Red neuronal 'bcakpropagation' para AND generada. Resultado: {red_neuronal_and}\n")

    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Hubo un error al generar la red neuronal AND: {e}\n")
        
    try:
        # Test case: 
        # - Entrada: backpropagation_network(inputs=funcion_or, perceptrons=perceptrones, layers=capas)
        # - Salida esperada: float 0 ~ 1
        # - Propósito: Ejecución de la red neuronal aplicada en la función "OR"
        red_neuronal_or = forward_propagation_network(inputs=funcion_or, perceptrons=perceptrones, layers=capas)
        print(f"Red neuronal 'forward' para OR generada. Resultado: {red_neuronal_or}\n")  
        red_neuronal_or = backpropagation_network(inputs=funcion_or, perceptrons=perceptrones, layers=capas)
        print(f"Red neuronal 'bcakpropagation' para OR generada. Resultado: {red_neuronal_or}\n")    
    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Hubo un error al generar la red neuronal OR: {e}\n")
        
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status

if __name__ == "__main__":
    sys.exit(main())