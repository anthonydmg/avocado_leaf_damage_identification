from PIL import Image

# Ruta de la imagen original
input_path = "20251021_103028.jpg"

# Ruta donde se guardará la nueva imagen
output_path = "imagen_guardada.jpg"

# Leer la imagen
imagen = Image.open(input_path)

# (Opcional) Mostrar información básica
print(imagen.format, imagen.size, imagen.mode)

# Guardar la imagen (puede ser el mismo o diferente formato)
imagen.save(output_path)

print("Imagen guardada correctamente")


