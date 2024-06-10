import sys
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

def crear_dotplot(secuencia1, secuencia2, len_secuencia1, len_secuencia2):
    secuencia1_decoded = decode_sequence(secuencia1, len_secuencia1)
    secuencia2_decoded = decode_sequence(secuencia2, len_secuencia2)

    dotplot = lil_matrix((len_secuencia1, len_secuencia2), dtype=np.uint8)
    for i in range(len_secuencia1):
        matches = secuencia1_decoded[i] == secuencia2_decoded
        dotplot[i, matches] = 1
    return dotplot

def main(file1, file2, limite, output):
    start_time = time.time()

    # Lectura y codificación de secuencias
    read_start = time.time()
    secuencia1, len_secuencia1 = merge_sequences_from_fasta(file1, limite)
    secuencia2, len_secuencia2 = merge_sequences_from_fasta(file2, limite)
    read_end = time.time()
    print(f"Tiempo de lectura y codificación de secuencias: {read_end - read_start} segundos")
    
    print(f"Secuencia 1 leída ({len_secuencia1} bases)")
    print(f"Secuencia 2 leída ({len_secuencia2} bases)")
    print(f"Total de bases usadas de secuencia 1: {len_secuencia1}")
    print(f"Total de bases usadas de secuencia 2: {len_secuencia2}")

    # Creación del dotplot
    dotplot_start = time.time()
    dotplot = crear_dotplot(secuencia1, secuencia2, len_secuencia1, len_secuencia2)
    dotplot_end = time.time()
    print(f"Tiempo de creación del dotplot: {dotplot_end - dotplot_start} segundos")
    
    # Guardar la imagen del dotplot
    plot_start = time.time()
    preview_size = 30000
    dotplot_preview = dotplot[:preview_size, :preview_size].toarray()
    
    plt.imshow(dotplot_preview, cmap='gray', aspect='auto')
    plt.title('Dotplot (Vista previa)')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig(output)
    plot_end = time.time()
    print(f"Tamaño de imagen: {preview_size} * {preview_size}")
    print(f"Tiempo de guardado de la imagen del dotplot: {plot_end - plot_start} segundos")
    
    end_time = time.time()
    print(f"\nEl código se ejecutó en: {end_time - start_time} segundos")
    print("La matriz resultado tiene un tamaño de " + str(sys.getsizeof(dotplot)) + " bytes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mi Aplicación Secuencial')
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
