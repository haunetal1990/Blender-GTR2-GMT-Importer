bl_info = {
    "name": "GTR2 GMT Importer",
    "author": "haunetal1990",
    "version": (0, 8, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > GTR2 (.gmt)",
    "description": "GTR2 GMT Importer",
    "category": "Import-Export",
}

import bpy
import struct
import os
import re
import traceback
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, CollectionProperty
from bpy.types import Operator, OperatorFileListElement

FMT_UINT = "=I"
FMT_3F   = "=fff"
FMT_H    = "=H"
VERTEX_STRIDE    = 32
UV_STREAM_STRIDE = 32
UV_OFFSET        = 8

# ============================================================
# Header / Geometry Parsing
# ============================================================

def find_submesh_headers(data):
    magic_values = [0x20004011, 0x20004015]
    positions = []
    for i in range(0, len(data) - 4, 4):
        val = struct.unpack_from(FMT_UINT, data, i)[0]
        if val in magic_values:
            positions.append(i)
    return positions

def parse_header(data, pos):
    return {
        'id':      struct.unpack_from(FMT_UINT, data, pos + 0x04)[0],
        'v_count': struct.unpack_from(FMT_UINT, data, pos + 0x08)[0],
        'v_off':   struct.unpack_from(FMT_UINT, data, pos + 0x0C)[0],
        'uv_off':  struct.unpack_from(FMT_UINT, data, pos + 0x14)[0],
        'i_count': struct.unpack_from(FMT_UINT, data, pos + 0x24)[0],
        'i_off':   struct.unpack_from(FMT_UINT, data, pos + 0x28)[0],
    }

def read_vertices(data, v_off, v_count):
    verts = []
    start = v_off + 4
    for i in range(v_count):
        pos = start + i * VERTEX_STRIDE
        if pos + 12 > len(data):
            break
        x, y, z = struct.unpack_from(FMT_3F, data, pos)
        verts.append((x, z, y))
    return verts

def is_sequential(data, i_off, i_count):
    start = i_off + 4
    for i in range(min(20, i_count)):
        pos = start + i * 2
        if pos + 2 > len(data):
            return False
        val = struct.unpack_from(FMT_H, data, pos)[0]
        if val != i:
            return False
    return True

def read_faces_sequential(v_count):
    faces = []
    for i in range(v_count // 3):
        faces.append((i*3, i*3+1, i*3+2))
    return faces

def read_faces_indexed(data, i_off, i_count, v_count):
    file_size = len(data)
    start = i_off + 4
    raw = []
    for i in range(i_count):
        pos = start + i * 2
        if pos + 2 > file_size:
            break
        val = struct.unpack_from(FMT_H, data, pos)[0]
        raw.append(val)
    
    faces = []
    for i in range(len(raw) // 3):
        i0 = raw[i*3 + 0]
        i1 = raw[i*3 + 1]
        i2 = raw[i*3 + 2]
        if i0 >= v_count or i1 >= v_count or i2 >= v_count:
            continue
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        faces.append((i0, i1, i2))
    return faces

def read_uvs(data, uv_off, v_count):
    file_size = len(data)
    uvs = []
    if uv_off == 0 or uv_off >= file_size:
        return [(0.0, 0.0)] * v_count
    
    potential = struct.unpack_from(FMT_UINT, data, uv_off)[0]
    start = uv_off + 4 if (potential == v_count or potential == 0) else uv_off
    
    for i in range(v_count):
        pos = start + i * UV_STREAM_STRIDE + UV_OFFSET
        if pos + 8 > file_size:
            uvs.append((0.0, 0.0))
            continue
        u, v = struct.unpack_from("=ff", data, pos)
        uvs.append((u, 1.0 - v))
    return uvs

# ============================================================
# Material / Texture Parsing
# ============================================================

RESERVED_WORDS = {
    'L0', 'L1', 'L2', 'L3', 'L4',
    'T0', 'T1', 'T2', 'T3', 'T4',
    'XT0', 'XT1', 'XT2', 'XT3',
    'DIFFUSE', 'SPECULAR', 'BUMP', 'CMAP',
    'NOLIGHTING', 'USEVERTEXONLY', 'USEVERTEX',
    'DIFFUSET0', 'SPECULART0', 'BUMPT0', 'CMAPT0',
}

def _is_shader_stage_name(name):
    if not name:
        return False
    if re.match(r'^L[0-9]', name):
        return True
    if name in RESERVED_WORDS:
        return True
    return False

def _is_valid_material_name(name):
    if not name or len(name) < 3:
        return False
    if '.' in name:
        return False
    if _is_shader_stage_name(name):
        return False
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return False
    return True

def find_material_texture_pairs(data, file_size):
    results = []
    stage_pattern = re.compile(
        rb'L([0-9])(?:DIFFUSE|SPECULAR|BUMP|CMAP|NOLIGHTING|USEVERTEXONLY|USEVERTEX)[A-Z0-9]*\x00'
    )
    
    stages = []
    for m in stage_pattern.finditer(data):
        stage_str = m.group(0).rstrip(b'\x00').decode('ascii', errors='replace')
        level = int(m.group(1))
        stages.append((m.start(), m.end(), stage_str, level))
    
    if not stages:
        return results
    
    material_groups = []
    current_group = [stages[0]]
    for i in range(1, len(stages)):
        prev_level = stages[i-1][3]
        curr_level = stages[i][3]
        gap = stages[i][0] - stages[i-1][1]
        
        if curr_level <= prev_level or gap > 4096:
            material_groups.append(current_group)
            current_group = [stages[i]]
        else:
            current_group.append(stages[i])
    material_groups.append(current_group)
    
    tex_pattern = re.compile(
        rb'([A-Za-z][A-Za-z0-9_\-]{1,40}\.(?:DDS|dds|TGA|tga|BMP|bmp))\x00'
    )
    name_pattern = re.compile(rb'([A-Za-z_][A-Za-z0-9_]{2,31})\x00')
    
    for group in material_groups:
        first_stage_start = group[0][0]
        last_stage_end = group[-1][1]
        
        search_start = max(0, first_stage_start - 512)
        chunk_before = data[search_start:first_stage_start]
        
        candidates = []
        for nm in name_pattern.finditer(chunk_before):
            candidate = nm.group(1).decode('ascii', errors='replace')
            if _is_valid_material_name(candidate):
                candidates.append((nm.start(), candidate))
        
        mat_name = candidates[-1][1] if candidates else None
        
        search_end = min(file_size, last_stage_end + 4096)
        chunk_after = data[last_stage_end:search_end]
        
        tex_name = None
        tex_match = tex_pattern.search(chunk_after)
        if tex_match:
            tex_name = tex_match.group(1).decode('ascii', errors='replace')
        
        results.append((mat_name, tex_name))
    
    return results

# ============================================================
# Material Creation (STABLE OPAQUE ONLY)
# ============================================================

def get_or_create_material(mat_name, tex_name, base_dir):
    if not mat_name:
        mat_name = "unnamed_material"
    
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    # Sicherstellen, dass das Material opak ist (Behebt Geister-Transparenzen)
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'OPAQUE'
    
    nt = mat.node_tree
    nt.nodes.clear()
    
    out  = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    out.location  = (600, 300)
    bsdf.location = (300, 300)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    if tex_name:
        tex_node = nt.nodes.new('ShaderNodeTexImage')
        tex_node.location = (0, 300)
        
        candidates = [tex_name, tex_name.upper(), tex_name.lower()]
        base, ext = os.path.splitext(tex_name)
        candidates.extend([
            base + ext.lower(),
            base + ext.upper(),
            base.upper() + ext.lower(),
            base.lower() + ext.lower(),
        ])
        
        loaded = False
        for candidate in candidates:
            path = os.path.join(base_dir, candidate)
            if os.path.exists(path):
                try:
                    existing = bpy.data.images.get(os.path.basename(path))
                    if existing:
                        tex_node.image = existing
                    else:
                        tex_node.image = bpy.data.images.load(path)
                    
                    loaded = True
                    break
                except Exception:
                    pass
        
        if loaded:
            nt.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                
    return mat

# ============================================================
# Main Import
# ============================================================

def read_gmt_file(filepath, context):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            
        file_size = len(data)
        base_dir  = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        ordered_mats = find_material_texture_pairs(data, file_size)
        headers = find_submesh_headers(data)
        
        if not headers:
            print(f"ERROR: no headers found in {base_name}.gmt")
            return {'CANCELLED'}
            
        all_verts   = []
        all_faces   = []
        all_uvs     = []
        all_mat_ids = []
        materials   = []
        
        for hdr_idx, hdr_pos in enumerate(headers):
            desc = parse_header(data, hdr_pos)
            
            if desc['v_count'] == 0 or desc['v_count'] > 100000:
                continue
            if desc['v_off'] == 0 or desc['v_off'] >= file_size:
                continue
                
            v_end = desc['v_off'] + 4 + desc['v_count'] * VERTEX_STRIDE
            if v_end > file_size:
                continue
                
            verts = read_vertices(data, desc['v_off'], desc['v_count'])
            uvs   = read_uvs(data, desc['uv_off'], desc['v_count'])
            
            if desc['i_count'] == 0 or desc['i_off'] == 0:
                if desc['v_count'] % 3 != 0:
                    continue
                faces = read_faces_sequential(desc['v_count'])
            else:
                seq = is_sequential(data, desc['i_off'], desc['i_count'])
                if seq:
                    if desc['v_count'] % 3 != 0:
                        continue
                    faces = read_faces_sequential(desc['v_count'])
                else:
                    faces = read_faces_indexed(data, desc['i_off'], desc['i_count'], desc['v_count'])
                    
            if not verts or not faces:
                continue
                
            sub_id = desc['id']
            mat_name = None
            tex_name = None
            
            if sub_id < len(ordered_mats):
                mat_name, tex_name = ordered_mats[sub_id]
            
            if not mat_name:
                if tex_name:
                    mat_name = os.path.splitext(tex_name)[0]
                else:
                    mat_name = base_name + "_mat_" + str(sub_id)
                    
            mat = get_or_create_material(mat_name, tex_name, base_dir)
            if mat not in materials:
                materials.append(mat)
                
            mat_slot_idx = materials.index(mat)
            vert_offset = len(all_verts)
            
            all_verts.extend(verts)
            all_uvs.extend(uvs)
            
            for face in faces:
                shifted = tuple(fi + vert_offset for fi in face)
                all_faces.append(shifted)
                all_mat_ids.append(mat_slot_idx)
                
        if not all_verts or not all_faces:
            print(f"ERROR: no geometry found in {base_name}.gmt")
            return {'CANCELLED'}
            
        mesh = bpy.data.meshes.new(base_name)
        obj  = bpy.data.objects.new(base_name, mesh)
        
        mesh.from_pydata(all_verts, [], all_faces)
        mesh.update()
        mesh.validate(verbose=False)
        
        for mat in materials:
            obj.data.materials.append(mat)
            
        uv_layer = mesh.uv_layers.new(name="UVMap")
        mesh.uv_layers.active = uv_layer
        
        for loop in mesh.loops:
            v_idx = loop.vertex_index
            if v_idx < len(all_uvs):
                uv_layer.data[loop.index].uv = all_uvs[v_idx]
                
        for poly_idx, poly in enumerate(mesh.polygons):
            poly.use_smooth = True
            if poly_idx < len(all_mat_ids):
                poly.material_index = all_mat_ids[poly_idx]
                
        context.collection.objects.link(obj)
        return {'FINISHED'}
        
    except Exception as e:
        print(f"ERROR beim Importieren von {os.path.basename(filepath)}: {str(e)}")
        traceback.print_exc()
        return {'CANCELLED'}

# ============================================================
# Blender Operator
# ============================================================

class ImportGTR2GMT(Operator, ImportHelper):
    bl_idname  = "import_scene.gtr2_gmt"
    bl_label   = "GTR2 GMT Import"
    bl_description = "Import an GTR2 GMT file as an object"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".gmt"
    filter_glob: StringProperty(default="*.gmt", options={'HIDDEN'}, maxlen=255)
    files: CollectionProperty(type=OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        for f in self.files:
            fp = os.path.join(self.directory, f.name)
            try:
                read_gmt_file(fp, context)
            except Exception as e:
                print(f"ERROR bei Datei {f.name}: {str(e)}")
                traceback.print_exc()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def menu_func_import(self, context):
    self.layout.operator(ImportGTR2GMT.bl_idname, text="GTR2 (.gmt)")

def register():
    bpy.utils.register_class(ImportGTR2GMT)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportGTR2GMT)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
