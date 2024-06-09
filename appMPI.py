import sys
from mpi4py import MPI
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse
import time

def merge_sequences_from_fasta(file_path):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq))
    return "".join(sequences)

def crear_dotplot_parcial(secuencia1, secuencia2, start, end):
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    dotplot_parcial = lil_matrix((end - start, len(secuencia2)), dtype=np.uint8)
    for i in range(start, end):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot_parcial[i - start, matches] = 1
    return dotplot_parcial

def main(file1, file2, limite, output):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        secuencia1 = merge_sequences_from_fasta(file1)[:limite]
        secuencia2 = merge_sequences_from_fasta(file2)[:limite]
        print(f"Proceso {rank}: secuencia 1 leída ({len(secuencia1)} bases)")
        print(f"Proceso {rank}: secuencia 2 leída ({len(secuencia2)} bases)")
        begin = time.time()
    else:
        secuencia1 = None
        secuencia2 = None
        begin = None

    secuencia1 = comm.bcast(secuencia1, root=0)
    secuencia2 = comm.bcast(secuencia2, root=0)
    begin = comm.bcast(begin, root=0)

    part_size = len(secuencia1) // size
    start = rank * part_size
    end = (rank + 1) * part_size if rank != size - 1 else len(secuencia1)

    dotplot_parcial = crear_dotplot_parcial(secuencia1, secuencia2, start, end)
    
    dotplot = None
    if rank == 0:
        dotplot = lil_matrix((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    
    dotplot_parcial = dotplot_parcial.tocsr()
    dotplot_parcial = comm.gather(dotplot_parcial, root=0)
    
    if rank == 0:
        for i, dp in enumerate(dotplot_parcial):
            dotplot[i * part_size: (i + 1) * part_size] = dp
        
        preview_size = 1000
        dotplot_preview = dotplot[:preview_size, :preview_size].toarray()

        plt.imshow(dotplot_preview, cmap='gray')
        plt.title('Dotplot (Vista previa)')
        plt.xlabel('Secuencia 2')
        plt.ylabel('Secuencia 1')
        plt.savefig(output)
        
        print(f"\n El código se ejecutó en: {time.time() - begin} segundos")
        print("la matriz resultado tiene un tamaño de " + str(sys.getsizeof(dotplot)) + " bytes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mi Aplicación MPI')
    parser.add_argument('--file1', type=str, help='Ruta del archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Umbral')
    parser.add_argument('--output', type=str, help='Archivo de salida')
    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2
    limite = args.limite
    output = args.output

    main(file1, file2, limite, output)
