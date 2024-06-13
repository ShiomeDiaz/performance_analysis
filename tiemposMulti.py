import argparse
import matplotlib.pyplot as plt

def calcular_metricas(tiempo_total, tiempo_procesamiento, num_procesos):
    secuencial = tiempo_procesamiento / num_procesos
    paralelizable = tiempo_procesamiento - secuencial
    tiempo_muerto = tiempo_total - (secuencial + paralelizable)
    speedup = tiempo_total / (secuencial + paralelizable)
    eficiencia = speedup / num_procesos
    return secuencial, paralelizable, tiempo_muerto, speedup, eficiencia

def plot_metricas(num_procesos, tiempos, speedups, eficiencias):
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(num_procesos, tiempos, marker='o')
    plt.title('Tiempos de Ejecución')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Tiempo (s)')

    plt.subplot(3, 1, 2)
    plt.plot(num_procesos, speedups, marker='o')
    plt.title('Aceleración (Speedup)')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Speedup')

    plt.subplot(3, 1, 3)
    plt.plot(num_procesos, eficiencias, marker='o')
    plt.title('Eficiencia')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Eficiencia')

    plt.tight_layout()
    plt.savefig('img/metricas.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calcular métricas de eficiencia y escalabilidad')
    parser.add_argument('--cores-list', nargs='+', type=int, required=True, help='Lista de números de procesos')

    args = parser.parse_args()
    cores_list = args.cores_list

    tiempos = []
    speedups = []
    eficiencias = []

    with open('txt/tiempos.txt', 'r') as f:
        lineas = f.readlines()
        tiempo_total = float(lineas[0].strip())
        tiempo_lectura = float(lineas[1].strip())
        tiempo_procesamiento = float(lineas[2].strip())
        tiempo_filtrado = float(lineas[3].strip())
        tiempo_creacion_imagen = float(lineas[4].strip())

    for num_procesos in cores_list:
        secuencial, paralelizable, tiempo_muerto, speedup, eficiencia = calcular_metricas(tiempo_total, tiempo_procesamiento, num_procesos)
        tiempos.append(tiempo_total)
        speedups.append(speedup)
        eficiencias.append(eficiencia)
        print(f"Número de procesos: {num_procesos}")
        print(f"Aceleración (Speedup): {speedup}")
        print(f"Eficiencia: {eficiencia}")

    plot_metricas(cores_list, tiempos, speedups, eficiencias)
