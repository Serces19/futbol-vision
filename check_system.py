#!/usr/bin/env python3
"""
Script para verificar que el sistema esté configurado correctamente
"""

import sys
import os
import importlib
from pathlib import Path
import cv2


def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando Python...")
    version = sys.version_info
    print(f"   Versión: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Se requiere Python 3.8 o superior")
        return False
    else:
        print("   ✅ Versión de Python compatible")
        return True


def check_dependencies():
    """Verificar dependencias principales"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('yaml', 'pyyaml'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    optional_packages = [
        ('psutil', 'psutil'),
        ('GPUtil', 'GPUtil'),
    ]
    
    all_good = True
    
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - REQUERIDO")
            all_good = False
    
    print("\n   Paquetes opcionales:")
    for module_name, package_name in optional_packages:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ⚠️  {package_name} - Opcional (para monitoreo avanzado)")
    
    return all_good


def check_cuda():
    """Verificar soporte CUDA"""
    print("\n🚀 Verificando CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"   ✅ CUDA disponible")
            print(f"   📱 Dispositivos: {device_count}")
            print(f"   🎯 Dispositivo actual: {device_name}")
            
            # Verificar memoria GPU
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            
            print(f"   💾 Memoria asignada: {memory_allocated:.2f} GB")
            print(f"   💾 Memoria reservada: {memory_reserved:.2f} GB")
            
        else:
            print("   ⚠️  CUDA no disponible - se usará CPU")
            
        # Verificar OpenCV CUDA
        opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if opencv_cuda:
            print(f"   ✅ OpenCV con soporte CUDA")
        else:
            print(f"   ⚠️  OpenCV sin soporte CUDA")
            
        return cuda_available
        
    except Exception as e:
        print(f"   ❌ Error verificando CUDA: {e}")
        return False


def check_models():
    """Verificar modelos disponibles"""
    print("\n🤖 Verificando modelos...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("   ❌ Directorio 'models' no encontrado")
        return False
    
    required_models = {
        "yolov8n-football.pt": "Detección de jugadores (ligero)",
        "yolov8n.pt": "Detección general",
        "SV_FT_TSWC_lines": "Líneas de campo",
        "SV_FT_TSWC_kp": "Puntos clave de campo"
    }
    
    optional_models = {
        "yolov8s.pt": "Detección más precisa",
        "SV_FT_WC14_lines": "Líneas alternativas",
        "SV_FT_WC14_kp": "Puntos clave alternativos"
    }
    
    models_found = 0
    total_required = len(required_models)
    
    print("   Modelos requeridos:")
    for model_name, description in required_models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            if model_path.is_file():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {model_name} - {description} ({size_mb:.1f} MB)")
            else:
                print(f"   ✅ {model_name} - {description} (Directorio)")
            models_found += 1
        else:
            print(f"   ❌ {model_name} - {description}")
    
    print("\n   Modelos opcionales:")
    for model_name, description in optional_models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            if model_path.is_file():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {model_name} - {description} ({size_mb:.1f} MB)")
            else:
                print(f"   ✅ {model_name} - {description} (Directorio)")
        else:
            print(f"   ⚠️  {model_name} - {description}")
    
    success_rate = models_found / total_required
    print(f"\n   📊 Modelos requeridos: {models_found}/{total_required} ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.75  # Al menos 75% de los modelos requeridos


def check_football_analytics():
    """Verificar que el módulo football_analytics funcione"""
    print("\n⚽ Verificando módulo football_analytics...")
    
    try:
        # Importar componentes principales
        from football_analytics.core import ConfigManager
        print("   ✅ ConfigManager")
        
        from football_analytics.core import setup_logging
        print("   ✅ Sistema de logging")
        
        from football_analytics.core import SystemMonitor
        print("   ✅ Sistema de monitoreo")
        
        from football_analytics.core.factory import DefaultComponentFactory
        print("   ✅ Factory de componentes")
        
        # Crear configuración de prueba
        config = ConfigManager()
        print("   ✅ Configuración por defecto")
        
        # Verificar que los paths de modelos sean válidos
        model_paths = config.model_paths
        print(f"   📁 Modelo de jugadores: {model_paths.yolo_player_model}")
        print(f"   📁 Modelo de balón: {model_paths.yolo_ball_model}")
        print(f"   📁 Modelo de líneas: {model_paths.field_lines_model}")
        print(f"   📁 Modelo de puntos clave: {model_paths.field_keypoints_model}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error importando football_analytics: {e}")
        return False


def check_test_videos():
    """Buscar videos de prueba"""
    print("\n🎥 Buscando videos de prueba...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    videos = []
    
    # Buscar en directorio actual
    for ext in video_extensions:
        videos.extend(Path('.').glob(f'*{ext}'))
    
    if videos:
        print(f"   ✅ Encontrados {len(videos)} videos:")
        for video in videos[:5]:  # Mostrar solo los primeros 5
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"      📹 {video.name} ({size_mb:.1f} MB)")
        
        if len(videos) > 5:
            print(f"      ... y {len(videos) - 5} más")
            
        return True
    else:
        print("   ⚠️  No se encontraron videos de prueba")
        print("      Coloca un archivo de video (.mp4, .avi, etc.) en el directorio actual")
        return False


def check_disk_space():
    """Verificar espacio en disco"""
    print("\n💾 Verificando espacio en disco...")
    
    try:
        import shutil
        
        # Verificar espacio en directorio actual
        total, used, free = shutil.disk_usage('.')
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"   📊 Espacio total: {total_gb:.1f} GB")
        print(f"   📊 Espacio usado: {used_gb:.1f} GB")
        print(f"   📊 Espacio libre: {free_gb:.1f} GB")
        
        if free_gb < 1.0:
            print("   ⚠️  Poco espacio libre (se recomienda >1GB)")
            return False
        elif free_gb < 5.0:
            print("   ⚠️  Espacio limitado (se recomienda >5GB para videos largos)")
            return True
        else:
            print("   ✅ Espacio suficiente")
            return True
            
    except Exception as e:
        print(f"   ❌ Error verificando espacio: {e}")
        return False


def generate_system_report():
    """Generar reporte del sistema"""
    print("\n📋 Generando reporte del sistema...")
    
    try:
        import platform
        import datetime
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0]
            }
        }
        
        # Agregar información de CUDA si está disponible
        try:
            import torch
            if torch.cuda.is_available():
                report["cuda"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name()
                }
            else:
                report["cuda"] = {"available": False}
        except:
            report["cuda"] = {"available": False, "error": "torch not available"}
        
        # Guardar reporte
        import json
        with open("system_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("   ✅ Reporte guardado en 'system_report.json'")
        return True
        
    except Exception as e:
        print(f"   ❌ Error generando reporte: {e}")
        return False


def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN DEL SISTEMA DE ANÁLISIS DE FÚTBOL")
    print("=" * 60)
    
    checks = [
        ("Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("CUDA", check_cuda),
        ("Modelos", check_models),
        ("Football Analytics", check_football_analytics),
        ("Videos de prueba", check_test_videos),
        ("Espacio en disco", check_disk_space),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"   ❌ Error en verificación de {check_name}: {e}")
            results[check_name] = False
    
    # Generar reporte
    generate_system_report()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
    
    print(f"\nResultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE CONFIGURADO!")
        print("Puedes ejecutar las pruebas con:")
        print("   python test_video_complete.py")
        print("   python test_video_cli.py")
        return 0
    elif passed >= total * 0.75:
        print("\n⚠️  SISTEMA MAYORMENTE CONFIGURADO")
        print("Algunas funciones pueden no estar disponibles")
        print("Puedes intentar ejecutar las pruebas básicas")
        return 0
    else:
        print("\n❌ SISTEMA NO CONFIGURADO CORRECTAMENTE")
        print("Revisa los errores arriba y corrige los problemas")
        return 1


if __name__ == "__main__":
    sys.exit(main())