import sys
import os

class HiddenPrints:
    def __enter__(self):
        #pass
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        #pass
        sys.stdout.close()
        sys.stdout = self._original_stdout
