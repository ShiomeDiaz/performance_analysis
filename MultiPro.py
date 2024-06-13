import multiprocessing
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO
import time
from multiprocessing import Manager
import os

# Lectura archivos
def lectura_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)

def crear_dotplot(arg):
    secuencia, secuencia2, indice = arg
    cod_secuencia = np.frombuffer(secuencia.encode(), dtype=np.uint8)
    cod_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)
    dotplot = np.equal.outer(cod_secuencia, cod_secuencia2).astype(np.uint8)
    return (indice, dotplot)

def dividir_secuencias(secuencia, numPartes):
    longitud_subsecuencia = len(secuencia) // numPartes
    subsecuencias = []
    for i in range(numPartes):
        inicio = i * longitud_subsecuencia
        fin = (i + 1) * longitud_subsecuencia if i < numPartes - 1 else len(secuencia)
        subsecuencia = secuencia[inicio:fin]
        subsecuencias.append(subsecuencia)
    return subsecuencias

def procesar_comparacion(secuencia, secuencia2, numProcesos):
    manager = Manager()
    dotplot = manager.list()

    pool = multiprocessing.Pool(processes=numProcesos)
    subSecuencia1 = dividir_secuencias(secuencia, numProcesos)
    resultados = [pool.apply_async(crear_dotplot, args=((subseq, secuencia2, i),)) 
                  for i, subseq in enumerate(subSecuencia1)]
    
    for i, resultado in tqdm(enumerate(resultados), total=numProcesos):
        indice, resultado_parcial = resultado.get()
        dotplot.append((indice, resultado_parcial))
    
    dotplot = sorted(dotplot, key=lambda x: x[0])

    dotplot_final = np.zeros((len(secuencia), len(secuencia2)), dtype=np.uint8)
    for indice, resultado_parcial in dotplot:
        inicio = indice * len(secuencia) // numProcesos
        fin = (indice + 1) * len(secuencia) // numProcesos
        dotplot_final[inicio:fin] = resultado_parcial
    
    pool.close()
    pool.join()

    return dotplot_final

def peso_matriz(matriz):
    bytes_matriz = sys.getsizeof(matriz)
    mb_matriz = bytes_matriz / (1024 ** 2)
    return mb_matriz

def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.imshow(matrix, cmap='gray', aspect='auto')
    plt.axis('off')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def detectar_lineas(args):
    dotplot_submatrix, indice = args
    filtro_mat = np.zeros_like(dotplot_submatrix)
    size = min(dotplot_submatrix.shape[0], dotplot_submatrix.shape[1])
    
    for i in range(size - 1):
        if dotplot_submatrix[i, i] == 1 and dotplot_submatrix[i + 1, i + 1] == 1:
            filtro_mat[i, i] = 1
            filtro_mat[i + 1, i + 1] = 1

    return (indice, filtro_mat)

def dividirMatriz(matriz, numPartes):
    longSubmatriz = matriz.shape[0] // numPartes
    submatrices = []
    for i in range(numPartes):
        inicio = i * longSubmatriz
        fin = (i + 1) * longSubmatriz if i < numPartes - 1 else matriz.shape[0]
        submatriz = matriz[inicio:fin]
        submatrices.append(submatriz)
    return submatrices

def filtrar_img(dotplot, numProcesos):
    manager = Manager()
    matrizFiltrada = manager.list()

    pool = multiprocessing.Pool(processes=numProcesos)
    subsecuencias_dotplot = dividirMatriz(dotplot, numProcesos)
    resultados = [pool.apply_async(detectar_lineas, args=((submatrix, i),)) for i, submatrix in enumerate(subsecuencias_dotplot)]

    for i, resultado in tqdm(enumerate(resultados), total=numProcesos):
        indice, resultado_parcial = resultado.get()
        matrizFiltrada.append((indice, resultado_parcial))

    matrizFiltrada = sorted(matrizFiltrada, key=lambda x: x[0])

    dotplotFiltrado = np.zeros_like(dotplot)
    for indice, resultado_parcial in matrizFiltrada:
        inicio = indice * dotplot.shape[0] // numProcesos
        fin = (indice + 1) * dotplot.shape[0] // numProcesos
        dotplotFiltrado[inicio:fin] = resultado_parcial

    pool.close()
    pool.join()

    return dotplotFiltrado

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enfoque Multiprocessing')
    parser.add_argument('--file1', type=str, help='Ruta archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Límite de la secuencia permitido')
    parser.add_argument('--cores', type=int, help='Número de procesos')

    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2
    limite = args.limite
    cores = args.cores

    if not os.path.exists('img'):
        os.makedirs('img')

    inicioTotal = time.time()
    inicioLectura = time.time()
    secuencia = lectura_fasta(file1)
    secuencia2 = lectura_fasta(file2)
    finLectura = time.time()

    numProcesos = cores
    
    inicioProcesamiento = time.time()
    dotplot = procesar_comparacion(secuencia[:limite], secuencia2[:limite], numProcesos)
    finProcesamiento = time.time()

    inicioFiltrado = time.time()
    dotplotFiltrado = filtrar_img(dotplot, numProcesos)
    finFiltrado = time.time()

    inicioCreacionImg = time.time()

    previewSize = 30000
    dotplot_pre = dotplot[:previewSize, :previewSize]
    dotplotFiltrado_pre = dotplotFiltrado[:previewSize, :previewSize]

    draw_dotplot(dotplot_pre, 'img/Multiprocessing.svg')
    draw_dotplot(dotplotFiltrado_pre, 'img/Multiprocessing_filtrado.svg')
    draw_dotplot(dotplot_pre[:500, :500], 'img/Multiprocessing_aumentada.svg')
    draw_dotplot(dotplotFiltrado_pre[:500, :500], 'img/Multiprocessing_filtrado_aumentada.svg')

    finCreacionImg = time.time()
    tiempoCreacionImagen = finCreacionImg - inicioCreacionImg

    finTotal = time.time()

    tiempoTotal = finTotal - inicioTotal 
    tiempoLectura = finLectura - inicioLectura
    tiempoProcesamiento = finProcesamiento - inicioProcesamiento
    tiempoFiltrado = finFiltrado - inicioFiltrado

    print("Tiempo total de ejecución:", tiempoTotal, "segundos")
    print("Tiempo de lectura de archivos:", tiempoLectura, "segundos")
    print("Tiempo de procesamiento:", tiempoProcesamiento, "segundos")
    print("Tiempo de filtrado:", tiempoFiltrado, "segundos")
    print("Tiempo de creación de imágenes:", tiempoCreacionImagen, "segundos")
    print("El tamaño de la matriz es:", dotplot.shape)
    print("La matriz resultado tiene un tamaño de", peso_matriz(dotplot), "Mb")
    print("La matriz filtrada tiene un tamaño de", peso_matriz(dotplotFiltrado), "Mb")

    # Guardar los tiempos en un archivo
    with open('txt/tiempos.txt', 'w') as f:
        f.write(f"{tiempoTotal}\n")
        f.write(f"{tiempoLectura}\n")
        f.write(f"{tiempoProcesamiento}\n")
        f.write(f"{tiempoFiltrado}\n")
        f.write(f"{tiempoCreacionImagen}\n")


#python3 MultiPro.py --file1 data/Salmonella.fna --file2 data/E_coli.fna --limite 30000 --cores 4
#Seguido de ejeucar el codigo; ejecutaremos el siguiente para poder ver la graficas de eficiencia y aceleracion:
#python3 tiemposMulti.py --cores-list 1 2 4 8