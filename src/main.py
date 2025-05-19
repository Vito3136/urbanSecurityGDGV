import os
import numpy as np
from PIL import Image


'''def exe_to_image(file_path, width=256):
    with open(file_path, 'rb') as f:
        byte_data = f.read()

    byte_arr = np.frombuffer(byte_data, dtype=np.uint8)

    # Calcola l'altezza dell'immagine
    height = int(np.ceil(len(byte_arr) / width))

    # Padding se necessario
    padded_len = height * width
    byte_arr = np.pad(byte_arr, (0, padded_len - len(byte_arr)), 'constant', constant_values=0)

    # Reshape in immagine
    byte_arr = byte_arr.reshape((height, width))

    # Converti in immagine
    img = Image.fromarray(byte_arr.astype(np.uint8))

    return img


# Esempio di salvataggio
img = exe_to_image('esempio.exe')
img.save('esempio.png')'''


# Legge un file .exe e lo converte in bytearray
with open("/resources/goodware_dataset/0a464a3765ffc0c23cf47345bf1185426af8e6b5711e015ca18027afcac2f2e0.exe", "rb") as f:
    bytecode = f.read()

# Ora bytecode è una sequenza di byte (bytes object)
print(bytecode[:20])  # stampa i primi 20 byte