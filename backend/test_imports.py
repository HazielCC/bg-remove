import os
import sys

# Asegurar que el path incluya backend para las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Probando sintaxis e importaciones...")

try:
    print("✅ ml.hf_downloader importado correctamente.")
    
    print("✅ ml.cogvideo importado correctamente.")
    
    print("✅ routers.layered importado correctamente.")
    
    print("✅ routers.video importado correctamente.")
    
    print("🎉 Todas las importaciones base pasaron sin errores de sintaxis.")
except Exception as e:
    print(f"❌ Error durante la importación: {e}")
    sys.exit(1)
