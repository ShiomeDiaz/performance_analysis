import sys
import gc
from mpi4py import MPI
import numpy as np
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
    decoded_sequence = ''.join(bits_to_base[(encoded_sequence[i // 4] >> ((3 - (i % 4)) * 2)) & 0b11] for i in range(length))
    return decoded_sequence

def merge_sequences_from_fasta(file_path, limite):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        if len("".join(sequences)) >= limite:
            break
    merged_sequence = "".join(sequences)[:limite]
    print(f"Merged sequence from {file_path} (limited to {limite} bases): {merged_sequence[:100]}...")  # Print first 100 bases for brevity
    encoded_sequence = encode_sequence(merged_sequence)
    print(f"Encoded sequence from {file_path} (limited to {limite} bases): {encoded_sequence[:10]}...")  # Print first 10 encoded elements for brevity
    return encoded_sequence, len(merged_sequence)

def crear_dotplot_parcial(secuencia1, secuencia2, start, end, len_secuencia2):
    secuencia1_decoded = decode_sequence(secuencia1, end - start)
    secuencia2_decoded = decode_sequence(secuencia2, len_secuencia2)

    dotplot_parcial = np.zeros((end - start, len_secuencia2), dtype=bool)
    for i in range(end - start):
        matches = np.frombuffer(secuencia1_decoded[i].encode(), dtype=np.uint8) == np.frombuffer(secuencia2_decoded.encode(), dtype=np.uint8)
        dotplot_parcial[i, matches] = True
    return dotplot_parcial

def send_large_array(comm, array, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == root:
        shape = array.shape
        dtype = array.dtype
    else:
        shape = None
        dtype = None

    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if rank != root:
        array = np.zeros(shape, dtype=dtype)
    
    comm.Bcast(array, root=root)
    
    return array

def gather_large_array(comm, partial_array, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        gathered_data = np.zeros_like(partial_array)
    else:
        gathered_data = None

    comm.Gather(partial_array, gathered_data, root=root)
    
    return gathered_data

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

    secuencia1 = send_large_array(comm, secuencia1, root=0)
    secuencia2 = send_large_array(comm, secuencia2, root=0)
    len_secuencia1 = comm.bcast(len_secuencia1, root=0)
    len_secuencia2 = comm.bcast(len_secuencia2, root=0)
    begin = comm.bcast(begin, root=0)

    # Ajustar el tamaño de la parte para asegurar que sea manejable
    part_size = max(len_secuencia1 // (size * 10), 1000)
    num_parts = (len_secuencia1 + part_size - 1) // part_size  # Número de partes a procesar

    for part in range(rank, num_parts, size):
        start = part * part_size
        end = min(start + part_size, len_secuencia1)
        dotplot_parcial = crear_dotplot_parcial(secuencia1, secuencia2, start, end, len_secuencia2)

        # Enviar cada parte al proceso raíz
        gathered_data = gather_large_array(comm, dotplot_parcial, root=0)
        if rank == 0:
            if 'dotplot' not in locals():
                dotplot = np.zeros((len_secuencia1, len_secuencia2), dtype=bool)
            start_index = part * part_size
            end_index = min(start_index + part_size, len_secuencia1)
            dotplot[start_index:end_index, :] = np.logical_or(dotplot[start_index:end_index, :], gathered_data)
        
        # Limpiar memoria después de procesar cada chunk
        del dotplot_parcial
        gc.collect()

    if rank == 0:
        preview_size = min(1000, len_secuencia1, len_secuencia2)
        dotplot_preview = dotplot[:preview_size, :preview_size]

        plt.imshow(dotplot_preview, cmap='gray', aspect='auto')
        plt.title('Dotplot (Vista previa)')
        plt.xlabel('Secuencia 2')
        plt.ylabel('Secuencia 1')
        plt.savefig(output)
        
        print(f"\n El código se ejecutó en: {time.time() - begin} segundos")
        print("La matriz resultado tiene un tamaño de " + str(sys.getsizeof(dotplot)) + " bytes")

        # Limpiar memoria después del procesamiento final
        del dotplot
        gc.collect()

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
