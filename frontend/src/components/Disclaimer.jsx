/**
 * Aviso obligatorio en TODAS las pantallas (regla del proyecto).
 * Se renderiza en el encabezado fijo (sticky) para que sea
 * visible sin importar la pantalla o el desplazamiento.
 */
export default function Disclaimer() {
  return (
    <p className="disclaimer" role="note">
      <span aria-hidden="true">⚠️ </span>
      Herramienta educativa, no es diagnóstico médico
    </p>
  );
}
