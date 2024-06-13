import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from mpi4py import MPI
import argparse

def merge_sequences_from_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)

def crear_dotplot(args):
    secuencia1, secuencia2, indice = args
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i in range(len(secuencia1)):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot[i, matches] = 1
    return (indice, dotplot)

def dividir_secuencia(secuencia, chunk_size):
    subsecuencias = []
    start = 0
    while start < len(secuencia):
        end = min(start + chunk_size, len(secuencia))
        subsecuencia = secuencia[start:end]
        subsecuencias.append(subsecuencia)
        start = end
    return subsecuencias

def calcular_peso_matriz(matriz):
    bytes_matriz = sys.getsizeof(matriz)
    megabytes_matriz = bytes_matriz / (1024 ** 2)
    return megabytes_matriz

def procesar_comparacion(secuencia1, secuencia2, chunk_size):
    inicio_parcial = time.time()
    comm = MPI.COMM_WORLD
    num_procesos = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        subsecuencias1 = dividir_secuencia(secuencia1, chunk_size)
    else:
        subsecuencias1 = None

    subsecuencias1 = comm.bcast(subsecuencias1, root=0)
    subsecuencia1 = subsecuencias1[rank]
    resultado_parcial = crear_dotplot((subsecuencia1, secuencia2, rank))
    resultados = comm.gather(resultado_parcial, root=0)

    if rank == 0:
        dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
        for resultado in resultados:
            indice, resultado_parcial = resultado
            inicio = indice * chunk_size
            fin = min(inicio + chunk_size, len(secuencia1))
            dotplot[inicio:fin] = resultado_parcial
        fin_tiempo_parcial = time.time()
        print("Tiempo parcial: ", fin_tiempo_parcial - inicio_parcial)
        return dotplot
    else:
        return None

def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='gray', aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.savefig(fig_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mi Aplicación MPI')
    parser.add_argument('--file1', type=str, help='Ruta del archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Numero de procesos')
    
    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2
    limite = args.limite
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    secuencia1 = merge_sequences_from_fasta(file1)
    secuencia2 = merge_sequences_from_fasta(file2)
    seccion_matriz = limite
    chunk_size = 1000

    if rank == 0:
        num_procesos = comm.Get_size()
        inicio_tiempo = time.time()
        dotplot = procesar_comparacion(secuencia1[:seccion_matriz], secuencia2[:seccion_matriz], chunk_size)
        preview_size = 30000
        dotplot_preview = dotplot[:preview_size, :preview_size]
        plt.imshow(dotplot_preview, cmap='gray')
        plt.title('Dotplot')
        plt.xlabel('Secuencia 2')
        plt.ylabel('Secuencia 1')
        plt.savefig('img/MPI')
        draw_dotplot(dotplot_preview, 'img/MPI.svg')
        draw_dotplot(dotplot_preview[:500, :500], 'img/MPI_aumentada.svg')

        fin_tiempo = time.time()
        tiempo_ejecucion = fin_tiempo - inicio_tiempo

        print("El código se ejecutó en:", tiempo_ejecucion, "segundos")
        print("El tamaño de la matriz es:", dotplot.shape)
        print("La matriz resultado tiene un tamaño de " + str(calcular_peso_matriz(dotplot)) + " Mb")
    else:
        procesar_comparacion(secuencia1[:seccion_matriz], secuencia2[:seccion_matriz], chunk_size)

#mpirun -n 4 python appMPI.py --file1=data/Salmonella.fna --file2=data/Salmonella.fna --limite=110000
