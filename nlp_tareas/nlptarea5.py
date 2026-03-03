#
##  tarea_5.py
### Borja Ramos Oliva

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("*" * 50)

### 1 - Pedimos y procesamos el texto
texto = input("\nIngresa un texto para procesar:\n> ")

if not texto.strip():
    print("Error: el texto no puede estar vacío.")
else:
    ###2 - En minúsculas
    texto_minusculas = texto.lower()
    print("\n*** Procesando texto ***\n")
    print(f"1. TEXTO ORIGINAL:\n{texto}\n")

    # mostramos mensajes de estado, no el contenido completo
    print("2. Pasando a minúsculas...")
    print("3. Eliminando puntuación...")
    print("4. Tokenizando...")
    print("5. Aplicando stemming...")
    print("6. Aplicando lematización...")
    print("7. Eliminando stopwords...\n")

    ### 3 - Eliminar puntuación
    texto_sin_puntuacion = texto_minusculas.translate(
        str.maketrans('', '', string.punctuation)
    )

    ### 4 - Tokenización
    tokens = word_tokenize(texto_sin_puntuacion)

    ### 5 - Stemming
    stemmer = PorterStemmer()
    tokens_stemming = [stemmer.stem(tok) for tok in tokens]

    ### 6 - Lematización
    lemmatizer = WordNetLemmatizer()
    tokens_lematizacion = [lemmatizer.lemmatize(tok) for tok in tokens]

    ### 7 - Eliminamos stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens_sin_stopwords = [
        tok for tok in tokens_lematizacion if tok not in stop_words
    ]

    ### 8 - Predicción de la siguiente palabra usando lemas
    print("\n*** PREDICCIÓN ***\n")
    print("Escribe una palabra lematizada para predecir la siguiente.")
    print("Escribe /fin para terminar.\n")

    while True:
        palabra_usuario = input("> ").lower().strip()

        # Opción de salida
        if palabra_usuario == "/fin":
            print("Fin de la predicción.\n")
            break

        # Cadena vacía
        if not palabra_usuario:
            print("No has escrito ninguna palabra. Prueba de nuevo o escribe /fin para salir.\n")
            continue

        # Comprobamos si la palabra está en el texto lematizado
        if palabra_usuario not in tokens_lematizacion:
            print(
                f"La palabra '{palabra_usuario}' no aparece en el texto procesado.\n"
                "Prueba con otra palabra que sí esté en el texto."
            )
            # Opcional: lista de palabras disponibles
            print(f"Palabras disponibles: {sorted(set(tokens_lematizacion))}\n")
            continue

        # Creamos los pares (bigramas) sobre los tokens lematizados
        pares = list(zip(tokens_lematizacion[:-1], tokens_lematizacion[1:]))
        siguientes = [b for (a, b) in pares if a == palabra_usuario]

        if siguientes:
            contador = Counter(siguientes)
            max_freq = max(contador.values())
            mas_probables = [pal for pal, freq in contador.items() if freq == max_freq]

            print(f"\nPalabra introducida: '{palabra_usuario}'")

            if len(mas_probables) == 1:
                print(
                    f"Palabra siguiente más probable: '{mas_probables[0]}' "
                    f"(aparece {max_freq} vez/veces)\n"
                )
            else:
                print(
                    "Palabras siguientes más probables (empate): "
                    + ", ".join(f"'{p}'" for p in mas_probables)
                    + f" (todas con frecuencia {max_freq})\n"
                )
            break
        else:
            print(f"No hay palabra siguiente para '{palabra_usuario}' en el texto.\n")
            break

    ### 9 - Resumen final en orden
    print("\n*** RESUMEN FINAL DEL TEXTO PROCESADO ***\n")
    print(f"1) Texto en minúsculas:\n{texto_minusculas}\n")
    print(f"2) Sin puntuación:\n{texto_sin_puntuacion}\n")
    print(f"3) Tokens:\n{tokens}\n")
    print(f"4) Tokens con stemming:\n{tokens_stemming}\n")
    print(f"5) Tokens lematizados:\n{tokens_lematizacion}\n")
    print(f"6) Tokens sin stopwords:\n{tokens_sin_stopwords}\n")

    print("*" * 50)

##
# Fin del programa