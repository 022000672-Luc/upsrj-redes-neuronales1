# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Redes Neuronales
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC03
# Archivo: test_main.py
# Descripción: Archivo de pruebas unitarias para validar el comportamiento de funciones del proyecto
# ============================================================
import sys, os, unittest, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
from forward_propagation.forward_propagation import forward_propagation_network

# Colores ANSI
GREEN = "\033[92m"
RED = "\033[91m"
LIGHT_RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"
SEPARATOR = f"{BOLD}{'='*50}{RESET}"

class CustomTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append((test))

class CustomTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Entradas de prueba
        self.funcion_and = np.array([0, 0, 0, 1])
        self.funcion_or = np.array([0, 1, 1, 1])
        self.perceptrones = 100
        self.capas = 100

    def test_forward_propagation_and(self):
        """Prueba la red neuronal con la función AND"""
        resultado = forward_propagation_network(
            inputs=self.funcion_and,
            perceptrons=self.perceptrones,
            layers=self.capas
        )
        self.assertIsInstance(resultado, float)
        self.assertGreaterEqual(resultado, 0.0)
        self.assertLessEqual(resultado, 1.0)
        
    def test_forward_propagation_or(self):
        """Prueba la red neuronal con la función OR"""
        resultado = forward_propagation_network(
            inputs=self.funcion_or,
            perceptrons=self.perceptrones,
            layers=self.capas
        )
        self.assertIsInstance(resultado, float)
        self.assertGreaterEqual(resultado, 0.0)
        self.assertLessEqual(resultado, 1.0)
        
    def test_forward_propagation_variability(self):
        """Verifica que la función responda a cambios en la entrada"""
        entrada_1 = np.array([0, 0])
        entrada_2 = np.array([1, 1])
        resultado_1 = forward_propagation_network(entrada_1, 10, 5)
        resultado_2 = forward_propagation_network(entrada_2, 10, 5)
        self.assertNotEqual(resultado_1, resultado_2, f"la función no responde a cambios en la entrada.")

    def test_forward_propagation_randomness(self):
        """Verifica que la función no esté hardcodeada: misma entrada, múltiples ejecuciones"""
        entrada = np.array([1.0, 0.0, 1.0, 0.0])
        resultados = set()
        for _ in range(5):
            salida = forward_propagation_network(inputs=entrada, perceptrons=10, layers=3)
            resultados.add(round(salida, 5))
        self.assertGreater(len(resultados), 1, f"la función puede estar hardcodeada.")

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEvaluation)
    silent_stream = io.StringIO()
    runner = CustomTestRunner(stream=silent_stream, verbosity=0)
    result = runner.run(suite)


    print(f"{BOLD}EVALUACION{RESET}")
    # Resultados individuales
    print(SEPARATOR)
    print(f"{BOLD}Resultados individuales:{RESET}")
    for test_case in result.successes:
        print(f"{test_case._testMethodName}: {GREEN}{BOLD}PASSED{RESET}")

    for test_case, traceback in result.failures + result.errors:
        print(f"{test_case._testMethodName}: {RED}{BOLD}FAILED{RESET}")
        # Extraer solo el mensaje de la última línea del traceback
        last_line = traceback.strip().split('\n')[-1]
        mensaje = last_line.split(':')[-1].strip()
        print(f"- detalles: {LIGHT_RED}{mensaje}{RESET}")

    # Resumen final
    print(SEPARATOR)
    print(f"{BOLD}Resumen final:{RESET}")
    if result.wasSuccessful():
        print(f"{GREEN}{BOLD}SUCCESS:{RESET} Todos los tests pasaron correctamente.")
    else:
        print(f"{RED}{BOLD}FAILED:{RESET} Uno o más tests fallaron.")
    print(SEPARATOR)
