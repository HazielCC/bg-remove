import sys
import os

# Asegurar que el path incluya backend para las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Probando sintaxis e importaciones...")

try:
    import ml.hf_downloader
    print("✅ ml.hf_downloader importado correctamente.")
    
    import ml.wan_video
    print("✅ ml.wan_video importado correctamente.")
    
    import routers.layered
    print("✅ routers.layered importado correctamente.")
    
    import routers.video
    print("✅ routers.video importado correctamente.")
    
    print("🎉 Todas las importaciones base pasaron sin errores de sintaxis.")
except Exception as e:
    print(f"❌ Error durante la importación: {e}")
    sys.exit(1)
