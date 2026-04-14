# PrecalDNP 3250 — paquete para análisis en Codex

## Objetivo
Analizar por qué ARC-FIT se desvía cuando el detector ya encontró un bottom sólido, con foco en:
- construcción/predicción de `top`
- criterio de aceptación de `top`
- uso real de `top`, `bottom` y laterales dentro de ARC-FIT

## Archivos incluidos
### Logs
- `Pegado text.txt`
  - corrida real del pipeline con logs de `DnpFacePipeline3250` y `RimDetectorBlock3250`

### Debug visual — detector
- `RIM_OD_DETECTOR_ONLY_1776179011447.png`
- `RIM_OD_DETECTOR_ONLY_1776179129541.png`
- `RIM_OI_DETECTOR_ONLY_1776179011560.png`
- `RIM_OI_DETECTOR_ONLY_1776179129626.png`

### Debug visual — ARC-FIT
- `RIM_OD_ARCFIT_ONLY_1776179021069.png`
- `RIM_OD_ARCFIT_ONLY_1776179139227.png`
- `RIM_OI_ARCFIT_ONLY_1776179021169.png`
- `RIM_OI_ARCFIT_ONLY_1776179139289.png`

## Lectura rápida del problema
- `bottom` aparece fuerte y continuo en varias corridas.
- `top` parece entrar a veces como línea horizontal válida, pero no siempre coincide con el aro real.
- ARC-FIT a veces queda desplazado o deformado respecto del borde observado.
- Sospecha principal: `top` está entrando con demasiada autoridad y/o ARC-FIT no tiene suficiente freno lateral real.

## Qué revisar
1. Cómo se construye `topObserved`
2. Qué diferencia hay entre `topObserved` y `topUsed`
3. Qué condiciones hacen que `top` sea aceptado para ARC-FIT
4. Cuánto pesan realmente `top`, `bottom` y laterales en el ajuste
5. Si ARC-FIT está validando laterales de verdad o en la práctica está fitteando casi sólo con top+bottom
6. Si el perfil `FULL_RIM` cae y luego el pipeline termina demasiado permisivo en `RANURADO/PERFORADO`

## Entregable esperado
- diagnóstico corto
- causa raíz
- parche mínimo y seguro
- logs nuevos para ver claramente:
  - `topObserved`
  - `topEstimated`
  - `topAcceptedForArcFit`
  - motivo de aceptación/rechazo
  - cantidad/calidad de side guides
  - peso final que usa ARC-FIT
