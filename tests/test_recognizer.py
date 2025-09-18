import unittest
from src.models.recognizer import Recognizer

class TestRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = Recognizer()

    def test_detectar_forma(self):
        resultado = self.recognizer.detectar_forma()
        self.assertIsInstance(resultado, str)

if __name__ == '__main__':
    unittest.main()