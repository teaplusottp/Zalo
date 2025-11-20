#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
import os
import sys

print("=" * 60)
print("TESTING IMPORT PATHS")
print("=" * 60)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n📍 Current directory: {current_dir}")
print(f"📁 Contents: {os.listdir(current_dir)}")

# Setup paths
mobilesamv2_path = os.path.join(current_dir, "MobileSAMv2")
tinyvit_path = os.path.join(mobilesamv2_path, "tinyvit")
efficientvit_path = os.path.join(mobilesamv2_path, "efficientvit")

print(f"\n🔍 Checking paths...")
print(f"   MobileSAMv2: {os.path.exists(mobilesamv2_path)} -> {mobilesamv2_path}")
print(f"   tinyvit: {os.path.exists(tinyvit_path)} -> {tinyvit_path}")
print(f"   efficientvit: {os.path.exists(efficientvit_path)} -> {efficientvit_path}")

# Add to sys.path
for path in [mobilesamv2_path, tinyvit_path, efficientvit_path]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

print(f"\n✅ Added to sys.path")

# Test imports
print(f"\n🧪 Testing imports...\n")

tests = [
    ("torch", "PyTorch"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("PIL", "Pillow"),
    ("scipy", "SciPy"),
    ("filterpy", "FilterPy"),
    ("transformers", "Transformers"),
    ("mobilesamv2.promt_mobilesamv2", "ObjectAwareModel (YOLO)"),
    ("mobilesamv2", "MobileSAMv2"),
    ("tinyvit.tiny_vit", "TinyViT"),
    ("mobilesamv2.modeling", "SAM components"),
]

failed = []
for module_name, description in tests:
    try:
        __import__(module_name)
        print(f"✅ {description:40} - {module_name}")
    except ImportError as e:
        print(f"❌ {description:40} - {module_name}")
        print(f"   Error: {e}")
        failed.append((description, module_name, str(e)))

print("\n" + "=" * 60)
if failed:
    print(f"❌ FAILED: {len(failed)} import(s) failed")
    for desc, mod, err in failed:
        print(f"   - {desc} ({mod})")
else:
    print("✅ SUCCESS: All imports working!")
print("=" * 60)

# Detailed test
if not failed:
    print("\n🔬 Detailed import test...")
    try:
        from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
        from mobilesamv2 import SamPredictor
        from mobilesamv2.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
        from tinyvit.tiny_vit import TinyViT
        print("✅ All detailed imports successful!")
    except Exception as e:
        print(f"❌ Detailed import failed: {e}")
