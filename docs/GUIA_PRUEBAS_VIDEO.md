# 🏈 Guía Completa para Pruebas con Video

## 📋 Requisitos Previos

### 1. Modelos Necesarios
Verifica que tienes estos modelos en el directorio `models/`:

**Modelos YOLO (Detección de Jugadores/Balón):**
- ✅ `yolov8n-football.pt` - Modelo ligero para jugadores
- ✅ `yolov8n.pt` - Modelo general para balón
- ✅ `yolov8s.pt` - Modelo más preciso (opcional)

**Modelos de Campo (Calibración):**
- ✅ `SV_FT_TSWC_lines` - Detección de líneas de campo
- ✅ `SV_FT_TSWC_kp` - Detección de puntos clave
- ✅ `SV_FT_WC14_lines` - Modelo alternativo de líneas
- ✅ `SV_FT_WC14_kp` - Modelo alternativo de puntos clave

### 2. Video de Prueba
Necesitas un video de fútbol. Formatos soportados:
- `.mp4` (recomendado)
- `.avi`
- `.mov`
- `.mkv`
- `.wmv`

**Características recomendadas del video:**
- Resolución: 720p o 1080p
- Duración: 30 segundos a 2 minutos para pruebas
- Vista del campo completa o parcial
- Buena iluminación

## 🚀 Métodos de Prueba

### Método 1: Prueba Completa con Monitoreo (Recomendado)

```bash
python test_video_complete.py
```

**Qué hace:**
- ✅ Verifica modelos disponibles
- ✅ Configura logging y monitoreo completo
- ✅ Procesa el video con métricas detalladas
- ✅ Genera reportes de rendimiento
- ✅ Crea visualizaciones y estadísticas

**Salidas generadas:**
```
test_output/
├── processed_video_YYYYMMDD_HHMMSS.mp4  # Video procesado
├── reports/
│   ├── system_status.json                # Estado del sistema
│   ├── health_check.json                 # Chequeo de salud
│   ├── debug_report.json                 # Reporte de debug
│   └── session_report.json               # Reporte de sesión
└── analytics_data.json                   # Datos de análisis

test_logs/
├── football_analytics_YYYYMMDD_HHMMSS.log  # Log principal
├── errors_YYYYMMDD_HHMMSS.log              # Solo errores
├── performance_YYYYMMDD_HHMMSS.log         # Métricas de rendimiento
└── analytics_YYYYMMDD_HHMMSS.log           # Datos de análisis
```

### Método 2: Prueba Rápida con CLI

```bash
python test_video_cli.py
```

**Qué hace:**
- ✅ Busca videos automáticamente
- ✅ Ejecuta el CLI oficial
- ✅ Procesamiento básico
- ✅ Salida simple

**Salidas generadas:**
```
cli_test_output/
├── processed_video.mp4     # Video procesado
├── analytics_data.json     # Datos básicos
└── player_stats.csv        # Estadísticas de jugadores
```

### Método 3: CLI Manual (Máximo Control)

```bash
# Comando básico
python -m football_analytics.cli.main process video.mp4

# Comando completo con opciones
python -m football_analytics.cli.main process video.mp4 \
    --output resultado.mp4 \
    --export-data salida/ \
    --confidence 0.4 \
    --teams 2 \
    --device cuda \
    --verbose -v
```

## 🔧 Configuración de Modelos

### Verificar Modelos Disponibles
```bash
python -c "
import sys
sys.path.append('.')
from pathlib import Path

models_dir = Path('models')
print('Modelos disponibles:')
for model in models_dir.glob('*'):
    if model.is_file():
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f'  {model.name}: {size_mb:.1f} MB')
    else:
        print(f'  {model.name}: Directorio')
"
```

### Configuración Automática de Dispositivo
El sistema detecta automáticamente:
- **CUDA**: Si tienes GPU NVIDIA compatible
- **CPU**: Como fallback

Para forzar un dispositivo:
```bash
--device cuda    # Forzar GPU
--device cpu     # Forzar CPU
--device auto    # Detección automática (default)
```

## 📊 Interpretación de Resultados

### 1. Video Procesado
- **Bounding boxes**: Rectángulos alrededor de jugadores
- **IDs de tracking**: Números que siguen a cada jugador
- **Colores de equipo**: Diferentes colores para cada equipo
- **Trayectorias**: Líneas que muestran el movimiento

### 2. Métricas de Rendimiento

**En los logs verás:**
```
Frame 30: 15.2 FPS, 65.8ms/frame
Frame 60: 14.8 FPS, 67.6ms/frame
```

**Interpretación:**
- **FPS**: Frames por segundo procesados
- **ms/frame**: Tiempo de procesamiento por frame
- **>10 FPS**: Rendimiento bueno
- **5-10 FPS**: Rendimiento aceptable
- **<5 FPS**: Rendimiento lento

### 3. Reportes JSON

**system_status.json:**
```json
{
  "processing": {
    "total_frames": 300,
    "average_fps": 12.5,
    "processing_time": {
      "average": 0.08,
      "min": 0.05,
      "max": 0.15
    }
  },
  "resources": {
    "cpu": {"average": 45.2, "peak": 78.1},
    "memory": {"average": 62.3, "peak": 85.4}
  }
}
```

**health_check.json:**
```json
{
  "status": "healthy",  // healthy, warning, critical
  "issues": [],         // Lista de problemas detectados
  "uptime_seconds": 125.3
}
```

## 🐛 Solución de Problemas

### Error: "No se puede abrir el video"
```bash
# Verificar que el archivo existe
ls -la video.mp4

# Verificar formato con ffprobe
ffprobe video.mp4
```

### Error: "CUDA out of memory"
```bash
# Usar CPU en lugar de GPU
python test_video_complete.py --device cpu

# O reducir el tamaño del batch
# Editar config: max_tracking_objects = 20
```

### Error: "Model not found"
```bash
# Verificar modelos
ls -la models/

# Descargar modelos faltantes
# (Instrucciones específicas según el modelo)
```

### Rendimiento Lento
1. **Usar modelo más ligero**: `yolov8n-football.pt` en lugar de `yolov8s.pt`
2. **Reducir resolución**: Redimensionar video antes del procesamiento
3. **Limitar frames**: Procesar solo una parte del video
4. **Usar CPU**: Si la GPU es muy lenta

## 📈 Optimización de Rendimiento

### Para Videos Largos
```python
# En test_video_complete.py, cambiar:
if frame_count >= 300:  # Cambiar a más frames
    break
```

### Para Mejor Precisión
```python
# Aumentar confidence threshold
confidence_threshold=0.6  # En lugar de 0.4
```

### Para Más Equipos
```python
# Cambiar número de equipos
n_teams=3  # Para 3 equipos
```

## 🎯 Casos de Uso Específicos

### 1. Análisis Rápido (30 segundos)
```bash
python test_video_cli.py
# Seleccionar video corto
```

### 2. Análisis Detallado (Con métricas)
```bash
python test_video_complete.py
# Revisar todos los reportes generados
```

### 3. Procesamiento en Lote
```bash
for video in *.mp4; do
    python -m football_analytics.cli.main process "$video" \
        --output "processed_$video" \
        --export-data "results_$(basename $video .mp4)"
done
```

## 📝 Checklist de Prueba

Antes de ejecutar:
- [ ] Modelos descargados en `models/`
- [ ] Video de prueba disponible
- [ ] Espacio en disco suficiente (>1GB)
- [ ] Python y dependencias instaladas

Durante la ejecución:
- [ ] Monitor de recursos del sistema
- [ ] Revisar logs en tiempo real
- [ ] Verificar que no hay errores críticos

Después de la ejecución:
- [ ] Revisar video procesado
- [ ] Analizar métricas de rendimiento
- [ ] Verificar datos exportados
- [ ] Revisar reportes de salud

## 🆘 Contacto y Soporte

Si encuentras problemas:
1. Revisa los logs en `test_logs/`
2. Verifica el `health_check.json`
3. Consulta el `debug_report.json`
4. Ejecuta con `--verbose -vv` para más detalles