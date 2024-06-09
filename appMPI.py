from mpi4py import MPI
import numpy as np
import os

def read_fasta(file_path):
    """Lee una secuencia de un archivo FASTA."""
    with open(file_path, 'r') as file:
        # Ignora la primera línea (descripción) y lee la secuencia
        return ''.join(line.strip() for line in file if not line.startswith('>'))

def generate_and_write_dotplot(seq1, seq2, start, end, output):
    """Genera y escribe un dotplot parcial para una porción de la secuencia."""
    with open(output, 'a') as f:
        for i in range(start, end):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    f.write(f"{i}\t{j}\n")

def main(file1, file2, output):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Leer los archivos FASTA en el proceso root
    if rank == 0:
        seq1 = read_fasta(file1)
        seq2 = read_fasta(file2)
        print(f"Proceso {rank}: secuencia 1 leída ({len(seq1)} bases)")
        print(f"Proceso {rank}: secuencia 2 leída ({len(seq2)} bases)")
        # Crear o vaciar el archivo de salida
        open(output, 'w').close()
    else:
        seq1 = None
        seq2 = None

    # Broadcast de las secuencias a todos los procesos
    seq1 = comm.bcast(seq1, root=0)
    seq2 = comm.bcast(seq2, root=0)
    
    # Dividir el trabajo entre los procesos
    part_size = len(seq1) // size
    start = rank * part_size
    end = (rank + 1) * part_size if rank != size - 1 else len(seq1)
    
    # Cada proceso genera y escribe una parte del dotplot
    generate_and_write_dotplot(seq1, seq2, start, end, output)
    
    # Sincronizar los procesos
    comm.Barrier()
    
    if rank == 0:
        print(f"Proceso {rank}: dotplot generado y guardado en {output}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Aplicación para generar dotplots usando mpi4py')
    parser.add_argument('--file1', type=str, required=True, help='Archivo FASTA de la primera secuencia')
    parser.add_argument('--file2', type=str, required=True, help='Archivo FASTA de la segunda secuencia')
    parser.add_argument('--output', type=str, required=True, help='Archivo de salida para el dotplot')
    args = parser.parse_args()
    
    # Construir rutas completas a los archivos FASTA
    file1_path = os.path.join('data', args.file1)
    file2_path = os.path.join('data', args.file2)
    output_path = os.path.join('data', args.output)
    
    main(file1_path, file2_path, output_path)
