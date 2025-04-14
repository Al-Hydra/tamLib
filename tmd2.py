from .utils.PyBinaryReader.binary_reader import *
from .pzze import readPZZE
import numpy as np

class TMD2(BrStruct):

    def __init__(self) -> None:
        self.magic = 'tmd0'
        self.name = ""
        self.version = 521
        self.modelFlags = 0
        
        self.bones = []
        self.models = []
        self.indexTables = []
        self.textures = []
        
    def __br_read__(self, br: 'BinaryReader', file_name = "") -> None:

        self.magic = br.read_str(4)
        if self.magic != 'tmd0':
            raise ValueError(f"Invalid magic: {self.magic}")

        if file_name:
            self.name = file_name
        
        unk0 = br.read_uint16()
        self.modelFlags = br.read_uint16()
        unk1 = br.read_uint32()
        unk2 = br.read_uint16()
        self.frameCount = br.read_int16()
        self.boundingBox = br.read_float32(6)
        self.modelsOffset = br.read_uint64()
        unk3 = br.read_uint32()
        self.subMeshEntriesOffset = br.read_uint32()
        self.materialsOffset = br.read_uint32()
        self.shaderParamsOffset = br.read_uint32()
        self.namesOffset = br.read_uint64()
        self.subMeshOffset = br.read_uint32()
        self.trianglesOffset = br.read_uint32()
        self.texturesOffset = br.read_uint64()
        self.materialTexturesOffset = br.read_uint32()
        self.verticesOffset = br.read_uint32()
        self.unkOffset = br.read_uint64()
        self.modelCount = br.read_uint32()
        unk4 = br.read_uint64()
        self.subMeshEntriesCount = br.read_uint32()
        self.materialCount = br.read_uint32()
        self.shaderParamsCount = br.read_uint32()
        unk5 = br.read_uint64()
        self.subMeshCount = br.read_uint32()
        self.trianglesCount = br.read_uint32()
        self.textureCount = br.read_uint64()
        self.materialTextureCount = br.read_uint32()
        self.vertexCount = br.read_uint32()
        if self.modelFlags & 0x2000:
            self.indexTablesOffset = br.read_uint32()
            self.tableIndicesOffset = br.read_uint32()
            self.boneMatrixOffset = br.read_uint32()
            self.boneHierarchyOffset = br.read_uint32()
            self.indexTablesCount = br.read_uint32()
            self.tableIndicesCount = br.read_uint32()
            self.boneCount = br.read_uint32()
            self.totalBoneCount = br.read_uint32()
            self.extraBoneInfoOffset = br.read_uint32()
            self.boneHierarchyOffset2 = br.read_uint32()
        
        #Textures Info
        br.seek(self.texturesOffset, Whence.BEGIN)
        self.textures = br.read_struct(TMD2Texture, self.textureCount)
        
        #Material Textures Info
        br.seek(self.materialTexturesOffset, Whence.BEGIN)
        self.materialTextures = br.read_struct(TMD2MatTexture, self.materialTextureCount, self.textures)
        
        #Shader Params Info
        br.seek(self.shaderParamsOffset, Whence.BEGIN)
        self.shaderParams = [br.read_float32(2)[1] for _ in range(self.shaderParamsCount)]
        
        #Material Info
        br.seek(self.materialsOffset, Whence.BEGIN)
        self.materials = br.read_struct(TMD2Material, self.materialCount, self.materialTextures, self.shaderParams)

        
        def unpack_normals(v):
            # Strip 4th component and normalize XYZ
            f = ((v[..., :3].astype(np.float32) / 255.0) * 2.0) - 1.0
            norm = np.linalg.norm(f, axis=-1, keepdims=True)
            norm[norm == 0] = 1.0
            return f / norm
        
        # Define converters for specific attributes
        _attribute_converters = {
            'normal':      unpack_normals,
            'normal2':     unpack_normals,
            'tangent':     unpack_normals,
            'binormal':    unpack_normals,
            'color':       lambda v: v.astype(np.float32) / 255.0,
            'color2':      lambda v: v.astype(np.float32) / 255.0,
            'boneWeights': lambda v: v.astype(np.float32) / 255.0,
            'boneWeights2':lambda v: v.astype(np.float32) / 255.0,
        }

        # Determine vertex attributes based on flags
        vertex_attributes = []
        if self.modelFlags & 0x2:
            vertex_attributes.append(('position', 'f4', 3))

        if self.modelFlags & 0x400:
            vertex_attributes.append(('boneWeights', 'u1', 4))
            vertex_attributes.append(('boneIDs', 'u1', 4))

        if self.modelFlags & 0x8000:
            vertex_attributes.append(('boneWeights2', 'u1', 4))
            vertex_attributes.append(('boneIDs2', 'u1', 4))

        if self.modelFlags & 0x4:
            vertex_attributes.append(('normal', 'u1', 4))

        if self.modelFlags & 0x8:
            vertex_attributes.append(('tangent', 'u1', 4))
            vertex_attributes.append(('binormal', 'u1', 4))

        if self.modelFlags & 0x80:
            vertex_attributes.append(('color', 'u1', 4))

        if self.modelFlags & 0x100:
            vertex_attributes.append(('normal2', 'u1', 4))

        if self.modelFlags & 0x200:
            vertex_attributes.append(('color2', 'u1', 4))

        if self.modelFlags & 0x10:
            vertex_attributes.append(('uv', 'f4', 2))
        if self.modelFlags & 0x20:
            vertex_attributes.append(('uv2', 'f4', 2))
        if self.modelFlags & 0x40:
            vertex_attributes.append(('uv3', 'f4', 2))

        # Read vertex buffer
        br.seek(self.verticesOffset, Whence.BEGIN)
        npVertexAtt = np.dtype(vertex_attributes)
        self.rawVertices = br.read_structured_array(npVertexAtt, self.vertexCount)

        
        # Convert fields in bulk
        converted_attributes = {}
        for name, dtype, size in vertex_attributes:
            column = self.rawVertices[name]
            if name in _attribute_converters:
                column = _attribute_converters[name](column)
            converted_attributes[name] = column

        # Create and fill vertex objects
        self.vertices = [TMD2Vertex() for _ in range(self.vertexCount)]
        for i in range(self.vertexCount):
            vertex = self.vertices[i]
            for name in converted_attributes:
                setattr(vertex, name, list(converted_attributes[name][i]))

        #print(self.vertices[0].__dict__)

        # Read triangle info
        br.seek(self.trianglesOffset, Whence.BEGIN)
        if self.modelFlags & 0x800:
            self.triangles = [br.read_uint32(3) for _ in range(self.trianglesCount)]
        else:
            self.triangles = [br.read_uint16(3) for _ in range(self.trianglesCount)]


        if self.modelFlags & 0x2000:
            #read all indices for the index table
            br.seek(self.tableIndicesOffset, Whence.BEGIN)
            self.allIndices = [br.read_uint32() for i in range(self.tableIndicesCount)]
            
            #pass the indices list to the index table class and read the individual index tables
            br.seek(self.indexTablesOffset, Whence.BEGIN)
            self.indexTables = br.read_struct(TMD2IndexTable, self.indexTablesCount, self.allIndices)
            
            if not self.indexTables:
                genericIndexTable = TMD2IndexTable()
                genericIndexTable.indices = [i for i in range(self.boneCount)]
                self.indexTables = [genericIndexTable]
            
            # read skeleton info
            br.seek(self.boneHierarchyOffset, Whence.BEGIN)
            self.bones = br.read_struct(TMD2Bone, self.boneCount, self.namesOffset)
            
            #read the matrix info
            br.seek(self.boneMatrixOffset, Whence.BEGIN)
            for i in range(self.boneCount):
                self.bones[i].matrix = [br.read_float32(4) for _ in range(4)]
            
            br.seek(self.extraBoneInfoOffset, Whence.BEGIN)
            for i in range(self.boneCount):
                bone = self.bones[i]
                bone.extra = br.read_int16()
            
            #print(self.bones[0].__dict__)
        
        br.seek(self.subMeshEntriesOffset, Whence.BEGIN)
        self.submeshEntries = br.read_struct(TMD2SubmeshEntry,self.subMeshEntriesCount)
        
        br.seek(self.subMeshOffset, Whence.BEGIN)
        self.submeshes = br.read_struct(TMD2Submesh,self.subMeshCount, self.triangles, self.vertices)
        
        #read model info
        br.seek(self.modelsOffset, Whence.BEGIN)
        self.models = br.read_struct(TMD2Model, self.modelCount, self.indexTables, self.submeshEntries, self.submeshes, self.materials, self.namesOffset)
                
        
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        #magic
        br.write_str_fixed("tmd0",4)
        br.write_int16(0)
        br.write_uint16(self.modelFlags)
        br.write_uint16(200) #this has something to do with hair meshes
        br.write_uint16(self.version)
        br.write_uint16(0x28)
        br.write_int16(-1)
        br.write_float(self.boundingBox)
        
        #we have to write the rest of the file before writing the rest of the header
        mat_buffer = BinaryReader()
        param_buffer = BinaryReader()
        mattex_buffer = BinaryReader()
        tex_buffer = BinaryReader()
        
        mat_buffer.write_struct(TMD2Material,None, param_buffer, mattex_buffer)
        


class TMD2Model(BrStruct):
    def __init__(self) -> None:
        self.boundingBox = [0,0,0,0,0,0]
        self.meshes = []
        self.materials = []
        self.name = ""
    
    def __br_read__(self, br: 'BinaryReader', indexTables, entries, submeshes, materials, namesOffset) -> None:
        self.boundingBox = br.read_float32(6)
        self.entriesCount = br.read_uint16()
        self.unk0 =  br.read_uint16()
        self.entriesStart = br.read_uint32()
        self.nameOffset = br.read_int32()
        self.unk1 =  br.read_uint32()
        self.unk2 =  br.read_uint32()
        
        if namesOffset > 0 and self.nameOffset != -1:
            self.name = br.read_str_at_offset(namesOffset + self.nameOffset)
        else:
            self.name = f"model_{str(self.unk1)}"
        
        materialIdx = -1
        indexTable = indexTables[0]
        
        for entry in entries[self.entriesStart: self.entriesStart + self.entriesCount]:
            if entry.type == 96:
                self.materials.append(materials[entry.index])
                materialIdx += 1
            
            elif entry.type == 64:
                indexTable = indexTables[entry.index]
            elif entry.type == 48:
                mesh = submeshes[entry.index]
                mesh.materialIndex = materialIdx
                mesh.indexTable = indexTable.indices
                self.meshes.append(mesh)
            
    
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        pass


class TMD2Submesh(BrStruct):
    def __init__(self):
        self.triangles = []
        self.vertices = []
        self.materialIndex = 0
        self.trianglesCount = 0
        self.trianglesStart = 0
        self.indexTable = []
    
    
    def __br_read__(self, br, triangles, vertices):
        self.trianglesCount = br.read_uint32()
        self.trianglesStart = br.read_uint32()
        
        
        vertex_map = {}  # Maps global index -> local index

        for tris in triangles[self.trianglesStart : self.trianglesStart + self.trianglesCount]:
            local_tri = []
            for global_idx in tris:
                if global_idx not in vertex_map:
                    vertex_map[global_idx] = len(self.vertices)
                    self.vertices.append(vertices[global_idx])
                local_tri.append(vertex_map[global_idx])
            self.triangles.append(local_tri)
    
    def __br_write__(self, br, triangles_buffer):
        br.write_uint32(len(self.triangles))
        br.write_uint32(triangles_buffer.size())

class TMD2SubmeshEntry(BrStruct):
    def __init__(self):
        self.type = None
        self.index = -1
    
    def __br_read__(self, br, *args):
        self.index = br.read_uint8()
        self.type = br.read_uint8()


class TMD2Texture(BrStruct):
    def __init__(self) -> None:
        self.hash = 0
        self.index = 0
        self.width = 0
        self.height = 0
        self.format = 0
        self.data = b''
    
    def __br_read__(self, br: 'BinaryReader', *args) -> None:
        self.hash = br.read_uint32()
        self.index = br.read_uint16()
        self.width = br.read_uint16()
        self.height = br.read_uint16()
        self.format = br.read_uint16()
    
    
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        pass


class TMD2MatTexture(BrStruct):
    def __init__(self) -> None:
        self.textureHash = 0
        self.textureIndex = 0
        self.texture = None
        self.slot = 0

    def __br_read__(self, br: 'BinaryReader', textures) -> None:
        self.textureHash = br.read_uint32()
        self.textureIndex = br.read_uint16()
        self.texture = textures[self.textureIndex]
        self.unk1 = br.read_uint16()
        self.unk2 = br.read_uint16()
        self.unk3 = br.read_uint8()
        self.slot = br.read_uint8()

class TMD2Material(BrStruct):
    def __init__(self) -> None:
        self.hash = 0
        self.name = ''
        self.shaderID = ""
        self.textures = []
        self.shaderParams = []

    def __br_read__(self, br: 'BinaryReader', textures, params) -> None:
        self.hash = br.read_uint32()
        self.name = str(self.hash)
        self.shaderID = br.read_str(4)
        self.textureStartIndex = br.read_uint16()
        self.textureCount = br.read_uint16()
        self.textures = textures[self.textureStartIndex: self.textureStartIndex + self.textureCount]
        
        self.shaderParamsStartIndex = br.read_uint16()
        self.shaderParamsCount = br.read_uint16()
        self.unk = br.read_int32()
        self.shaderParams = params[self.shaderParamsStartIndex: self.shaderParamsStartIndex + self.shaderParamsCount]

    def __br_write__(self, br: 'BinaryReader', matTextures, params) -> None:
        br.write_uint32(self.hash)
        br.write_str_fixed(self.shaderID,4)
        br.write_uint16(len(matTextures))
        br.write_uint16(len(self.textures))
        #add textures from this material to the material textures list
        matTextures.extend(self.textures)
        
        br.write_uint16(len(params))
        br.write_uint16(len(self.shaderParams))
        #add params from this material to the shader params list
        matTextures.extend(self.self.shaderParams)

class TMD2Vertex:
    def __init__(self) -> None:
        self.position = [0.0, 0.0, 0.0]
        self.normal = [0.0, 0.0, 0.0]
        self.normal2 = [0.0, 0.0, 0.0]
        self.tangent = [0.0, 0.0, 0.0]
        self.binormal = [0.0, 0.0, 0.0]
        self.uv = [0.0, 0.0]
        self.uv2 = [0.0, 0.0]
        self.uv3 = [0.0, 0.0]
        self.uv4 = [0.0, 0.0]
        self.color = [0, 0, 0, 255]
        self.color2 = [0, 0, 0, 255]
        self.boneIDs = [0, 0, 0, 0]
        self.boneWeights = [0, 0, 0, 0]
        self.boneIDs2 = [0, 0, 0, 0]
        self.boneWeights2 = [0, 0, 0, 0]


class TMD2Bone(BrStruct):
    def __init__(self) -> None:
        self.name = ''
        self.parentIndex = 0
        self.tailPosition = [0.0, 0.0, 0.0]
        self.matrix = np.zeros((4, 4), dtype=np.float32)
        self.extra = -1

    def __br_read__(self, br: 'BinaryReader', namesOffset = -1) -> None:
        self.hash = br.read_uint32()
        self.tailPosition = br.read_float32(3)
        self.parentIndex = br.read_int32()
        
        self.unk1 = br.read_uint16()
        self.nameOffset = br.read_uint16()
        if namesOffset > 0 and self.nameOffset != -1:
            self.name = br.read_str_at_offset(namesOffset + self.nameOffset, encoding='utf-8')
        else:
            self.name = str(self.hash)


class TMD2IndexTable(BrStruct):
    def __init__(self):
        self.indices = []
        self.indicesCount = 0
    
    def __br_read__(self, br, tableIndices):
        
        self.indicesCount = br.read_uint32()
        self.startIndex = br.read_uint32()
        
        self.indices = tableIndices[self.startIndex: self.startIndex + self.indicesCount]
