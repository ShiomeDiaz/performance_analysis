import sys
from mpi4py import MPI
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse
import time

def encode_sequence(sequence):
    base_to_bits = {'A': 0b00, 'C': 0b01, 'G': 0b10, 'T': 0b11}
    encoded_sequence = np.zeros((len(sequence) + 3) // 4, dtype=np.uint8)
    for i, base in enumerate(sequence):
        pos = i // 4
        offset = (3 - (i % 4)) * 2
        encoded_sequence[pos] |= base_to_bits[base] << offset
    return encoded_sequence

def decode_sequence(encoded_sequence, length):
    bits_to_base = {0b00: 'A', 0b01: 'C', 0b10: 'G', 0b11: 'T'}
    decoded_sequence = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        pos = i // 4
        offset = (3 - (i % 4)) * 2
        bits = (encoded_sequence[pos] >> offset) & 0b11
        decoded_sequence[i] = bits
    return decoded_sequence

def merge_sequences_from_fasta(file_path, limite):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        if len("".join(sequences)) >= limite:
            break
    merged_sequence = "".join(sequences)[:limite]
    return encode_sequence(merged_sequence), len(merged_sequence)

def crear_dotplot_parcial(secuencia1, secuencia2, start, end, len_secuencia2):
    secuencia1_decoded = decode_sequence(secuencia1, end)
    secuencia2_decoded = decode_sequence(secuencia2, len_secuencia2)

    dotplot_parcial = lil_matrix((end - start, len_secuencia2), dtype=np.uint8)
    for i in range(end - start):
        matches = secuencia1_decoded[start + i] == secuencia2_decoded
        dotplot_parcial[i, matches] = 1
    return dotplot_parcial

def main(file1, file2, limite, output):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        secuencia1, len_secuencia1 = merge_sequences_from_fasta(file1, limite)
        secuencia2, len_secuencia2 = merge_sequences_from_fasta(file2, limite)
        print(f"Proceso {rank}: secuencia 1 leída ({len_secuencia1} bases)")
        print(f"Proceso {rank}: secuencia 2 leída ({len_secuencia2} bases)")
        print(f"Total de bases usadas de secuencia 1: {len_secuencia1}")
        print(f"Total de bases usadas de secuencia 2: {len_secuencia2}")
        begin = time.time()
    else:
        secuencia1 = None
        secuencia2 = None
        len_secuencia1 = None
        len_secuencia2 = None
        begin = None

    secuencia1 = comm.bcast(secuencia1, root=0)
    secuencia2 = comm.bcast(secuencia2, root=0)
    len_secuencia1 = comm.bcast(len_secuencia1, root=0)
    len_secuencia2 = comm.bcast(len_secuencia2, root=0)
    begin = comm.bcast(begin, root=0)

    part_size = len_secuencia1 // size
    start = rank * part_size
    end = (rank + 1) * part_size if rank != size - 1 else len_secuencia1

    dotplot_parcial = crear_dotplot_parcial(secuencia1, secuencia2, start, end, len_secuencia2)
    
    dotplot = None
    if rank == 0:
        dotplot = lil_matrix((len_secuencia1, len_secuencia2), dtype=np.uint8)
    
    dotplot_parcial = dotplot_parcial.tocsr()
    dotplot_parcial = comm.gather(dotplot_parcial, root=0)
    
    if rank == 0:
        for i, dp in enumerate(dotplot_parcial):
            dotplot[i * part_size: (i + 1) * part_size] = dp
        
        preview_size = 100000
        dotplot_preview = dotplot[:preview_size, :preview_size].toarray()

        plt.imshow(dotplot_preview, cmap='gray', aspect='auto')
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
