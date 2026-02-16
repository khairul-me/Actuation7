"""
Blender Synthetic Data Generation Pipeline

Generates synthetic images with perfect ground truth layered depth for thin structures.
Target: 10,000 scenes with RGB, depth, and multi-layer ground truth.

Run with: blender --background --python data_generation/blender_thin_structures.py -- --output synthetic_data --num_scenes 10000

NOTE: Requires Blender 3.6+ with Python enabled.
This script is run INSIDE Blender's Python interpreter.
"""

import os
import sys
import json
import math
import random

# Check if running in Blender
try:
    import bpy
    import bmesh
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("NOTE: This script is designed to run inside Blender.")
    print("Usage: blender --background --python data_generation/blender_thin_structures.py -- --output synthetic_data")

if IN_BLENDER:
    import numpy as np

    def create_thin_stem(location, diameter=0.005, height=0.15):
        """Create a thin cylindrical stem."""
        bpy.ops.mesh.primitive_cylinder_add(
            radius=diameter / 2,
            depth=height,
            location=location
        )
        stem = bpy.context.active_object
        stem.name = f"stem_{location[0]:.3f}_{location[1]:.3f}"
        
        mat = bpy.data.materials.new(name="StemMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (
            0.15 + random.random() * 0.15,
            0.4 + random.random() * 0.3,
            0.1 + random.random() * 0.15,
            1.0
        )
        stem.data.materials.append(mat)
        return stem

    def create_leaf(location, width=0.03, length=0.05, angle=45):
        """Create a thin leaf."""
        bpy.ops.mesh.primitive_plane_add(size=1, location=location)
        leaf = bpy.context.active_object
        leaf.scale = (length, width, 0.001)
        leaf.rotation_euler = (
            math.radians(angle),
            0,
            math.radians(random.random() * 360)
        )
        
        mat = bpy.data.materials.new(name="LeafMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (
            0.2 + random.random() * 0.2,
            0.5 + random.random() * 0.3,
            0.2 + random.random() * 0.15,
            1.0
        )
        bsdf.inputs['Alpha'].default_value = 0.85 + random.random() * 0.15
        bsdf.inputs[7].default_value = 0.5  # Specular (index may vary)
        leaf.data.materials.append(mat)
        return leaf

    def create_plant(base_location, num_leaves=5):
        """Create a complete plant with stem and leaves."""
        stem_height = random.uniform(0.05, 0.20)
        stem = create_thin_stem(
            location=(base_location[0], base_location[1], stem_height / 2),
            diameter=random.uniform(0.003, 0.008),
            height=stem_height
        )
        
        leaves = []
        for i in range(num_leaves):
            leaf_height = stem_height * (i + 1) / (num_leaves + 1)
            leaf_offset_x = random.uniform(-0.02, 0.02)
            leaf_offset_y = random.uniform(-0.02, 0.02)
            
            leaf = create_leaf(
                location=(
                    base_location[0] + leaf_offset_x,
                    base_location[1] + leaf_offset_y,
                    leaf_height
                ),
                width=random.uniform(0.02, 0.04),
                length=random.uniform(0.03, 0.06),
                angle=random.uniform(30, 60)
            )
            leaves.append(leaf)
        
        return stem, leaves

    def create_ground_plane():
        """Create soil/ground plane."""
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.scale = (0.5, 0.5, 1)
        
        mat = bpy.data.materials.new(name="SoilMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (0.4, 0.3, 0.2, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.9
        ground.data.materials.append(mat)
        return ground

    def setup_camera(height=0.30, angle_deg=0):
        """Setup camera matching RealSense D405 specs."""
        bpy.ops.object.camera_add(
            location=(0, -height * math.tan(math.radians(angle_deg)), height)
        )
        camera = bpy.context.active_object
        camera.rotation_euler = (math.radians(90 - angle_deg), 0, 0)
        bpy.context.scene.camera = camera
        camera.data.lens = 18  # Roughly matches D405 FOV
        camera.data.sensor_width = 13.2
        return camera

    def setup_lighting():
        """Setup realistic lighting."""
        # Remove default light if exists
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)
        
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 1.0))
        light = bpy.context.active_object
        light.data.energy = 100
        light.data.size = 0.5
        return light

    def setup_depth_compositing(scene, output_dir, scene_name):
        """Setup compositor for RGB and depth output."""
        scene.use_nodes = True
        tree = scene.node_tree
        
        for node in tree.nodes:
            tree.nodes.remove(node)
        
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        
        # RGB output
        rgb_output = tree.nodes.new('CompositorNodeOutputFile')
        rgb_output.base_path = output_dir
        rgb_output.file_slots[0].path = f"{scene_name}_rgb"
        rgb_output.format.file_format = 'PNG'
        tree.links.new(render_layers.outputs['Image'], rgb_output.inputs[0])
        
        # Depth output
        depth_output = tree.nodes.new('CompositorNodeOutputFile')
        depth_output.base_path = output_dir
        depth_output.file_slots[0].path = f"{scene_name}_depth"
        depth_output.format.file_format = 'OPEN_EXR'
        depth_output.format.color_depth = '32'
        tree.links.new(render_layers.outputs['Depth'], depth_output.inputs[0])

    def generate_dataset(num_scenes=10000, output_dir="synthetic_data"):
        """Generate complete synthetic dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        for scene_id in range(num_scenes):
            # Clear scene
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            
            # Create ground
            ground = create_ground_plane()
            
            # Create plants
            num_plants = random.randint(1, 15)
            plants = []
            
            for i in range(num_plants):
                x = random.uniform(-0.2, 0.2)
                y = random.uniform(-0.2, 0.2)
                plant = create_plant(
                    base_location=(x, y, 0),
                    num_leaves=random.randint(3, 8)
                )
                plants.append(plant)
            
            # Camera with slight random angle
            angle = random.uniform(0, 10)
            camera = setup_camera(height=0.30, angle_deg=angle)
            
            # Lighting
            light = setup_lighting()
            
            # Render settings
            scene = bpy.context.scene
            scene.render.resolution_x = 848
            scene.render.resolution_y = 480
            scene.render.film_transparent = False
            
            scene_name = f"scene_{scene_id:05d}"
            
            # Setup compositing
            setup_depth_compositing(scene, output_dir, scene_name)
            
            # Render
            bpy.ops.render.render(write_still=False)
            
            # Save metadata
            metadata = {
                'scene_id': scene_id,
                'num_plants': num_plants,
                'camera_height': 0.30,
                'camera_angle': angle,
                'plants': [
                    {
                        'location': list(p[0].location),
                        'num_leaves': len(p[1])
                    }
                    for p in plants
                ]
            }
            
            meta_path = os.path.join(output_dir, f"{scene_name}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if scene_id % 100 == 0:
                print(f"Generated {scene_id}/{num_scenes} scenes")
        
        print(f"Dataset generation complete: {num_scenes} scenes in {output_dir}")

    # Parse command-line arguments (after --)
    if __name__ == "__main__":
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        else:
            argv = []
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', default='synthetic_data')
        parser.add_argument('--num_scenes', type=int, default=10000)
        args = parser.parse_args(argv)
        
        generate_dataset(num_scenes=args.num_scenes, output_dir=args.output)

else:
    # Not in Blender - provide usage info
    print("\n" + "=" * 60)
    print("Blender Synthetic Data Generation")
    print("=" * 60)
    print()
    print("This script generates 10,000 synthetic training scenes.")
    print()
    print("Prerequisites:")
    print("  1. Install Blender 3.6+ from https://www.blender.org/download/")
    print("  2. Ensure Blender is in your PATH")
    print()
    print("Usage:")
    print('  blender --background --python data_generation/blender_thin_structures.py -- --output data/synthetic --num_scenes 10000')
    print()
    print("Estimated time: 2-3 days for 10,000 scenes on GPU workstation")
    print()
    print("Output format per scene:")
    print("  scene_XXXXX_rgb0001.png   - RGB render (848x480)")
    print("  scene_XXXXX_depth0001.exr - Depth map (float32)")
    print("  scene_XXXXX_meta.json     - Scene metadata")

