/**
 * Reducción de la señal para visualización.
 *
 * La API acepta hasta 2 000 000 de muestras; dibujarlas todas sería
 * ilegible y lento. Agrupamos en bloques de `paso` muestras y dibujamos
 * el promedio de cada bloque, con x = índice central del bloque (en
 * **índices de muestra originales**), así las zonas anómalas
 * (índices de muestra) se dibujan 1:1 sin transformación.
 */

export function downsampleSignal(signal, maxPuntos = 1500) {
  const total = signal.length;
  if (total === 0) return { data: [], paso: 1, total: 0 };
  if (total <= maxPuntos) {
    return {
      data: signal.map((valor, i) => ({ x: i, y: valor })),
      paso: 1,
      total,
    };
  }
  const paso = Math.ceil(total / maxPuntos);
  const data = [];
  for (let inicio = 0; inicio < total; inicio += paso) {
    const fin = Math.min(inicio + paso, total);
    let suma = 0;
    for (let i = inicio; i < fin; i += 1) suma += signal[i];
    data.push({ x: inicio + (fin - inicio) / 2, y: suma / (fin - inicio) });
  }
  return { data, paso, total };
}
