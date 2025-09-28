# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: input_data.py
# Descripción: definición de una clase que abstrae la entrada de un perceptrón
# ============================================================
import random
#############################################################################################################################
# Perceptrón de una sola neurona                                                                                            #
# Una neurona toma n entradas (x1, x2, ... xn), las multiplica por sus respectivos pesos (w1, w2, ...wn),                   #
# les suma un sesgo (b), y produce una salida (a) mediante una combinación lineal:                                          #
#                                                                                                                           #
# z = x1 * w1 + x2 * w2 + ... xn * wn + b                                                                                   #
# a = f(z)  # En este ejemplo, aplicamos la sigmoide como función de activación                                             #
#                                                                                                                           #
# Diagrama conceptual:                                                                                                      #
#   x1 ─┐                                                                                                                   #
#       │                                                                                                                   #
#   x2 ─┼─► [ Neurona ] ──► a                                                                                               #
#       │                                                                                                                   #
#      ...      ↑                                                                                                           #
#       │   (w1, w2, ... wn, b)                                                                                             #
#   xn ─┘                                                                                                                   #
#                                                                                                                           #
#                                                                                                                           #
# Este modelo es la base de redes más complejas. Ideal para introducir conceptos como pesos, sesgo y salida lineal.         #
#                                                                                                                           #
# NOTE: https://docs.python.org/3/tutorial/classes.html                                                                     #
#                                                                                                                           #
#############################################################################################################################

# Paso 1: Abstracción de entradas de una neurona.
#
# TODO: Define una clase "InputData" que contenga todos los elementos del diagrama conceptual referentes a la entrada de un perceptrón.
#
# NOTE: * Considera todos los procesos que ocurren dentro de un perceptrón:
#           - Inicialización aleatoria del peso
#           - Actualización de los pesos para propagación
#
#       * Recuerda que la entrada de un perceptrón contiene:
#           - un valor "x"
#           - un peso "w"
#
class InputData:
    def __init__(self, x: float):
        # Parametro de entrada de usuario
        self.x = x
        # Inicializacion aleatoria
        self.w = self.init_weight()
    
    def init_weight(self):
        return random.random()
    
    def update_weight(self, w: float):
        self.w = w