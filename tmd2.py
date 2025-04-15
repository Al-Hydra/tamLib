from .utils.PyBinaryReader.binary_reader import *
from .pzze import readPZZE
import numpy as np

class TMD2(BrStruct):

    def __init__(self) -> None:
        self.magic = 'tmd0'
        self.name = ""
        self.version = 0x209
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
            
            
            unkBoneInfoCount = 0
            br.seek(self.extraBoneInfoOffset, Whence.BEGIN)
            for i in range(self.boneCount):
                bone = self.bones[i]
                bone.extra = br.read_int16()
                if bone.extra != -1:
                    unkBoneInfoCount += 1
            
            br.align_pos(16)
            self.unkBoneInfo = [br.read_float32(3) for i in range(unkBoneInfoCount)]
            
            #print(self.bones[0].__dict__)
        
        br.seek(self.subMeshEntriesOffset, Whence.BEGIN)
        self.submeshEntries = br.read_struct(TMD2SubmeshEntry,self.subMeshEntriesCount)
        
        br.seek(self.subMeshOffset, Whence.BEGIN)
        self.submeshes = br.read_struct(TMD2Submesh,self.subMeshCount, self.triangles, self.vertices)
        
        #read model info
        br.seek(self.modelsOffset, Whence.BEGIN)
        self.models = br.read_struct(TMD2Model, self.modelCount, self.indexTables, self.submeshEntries, self.submeshes, self.materials, self.namesOffset)
        
        #read the unknown section
        br.seek(self.unkOffset, Whence.BEGIN)
        self.unkSections = br.read_struct(TMD2Unk, self.modelCount + 1)
                
        
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        #magic
        br.write_str_fixed("tmd0",4)
        br.write_int16(0)
        br.write_uint16(self.modelFlags)
        br.write_uint16(200) #this has something to do with hair meshes
        br.write_uint16(self.version)
        br.write_uint16(0x28)
        br.write_int16(-1)
        br.write_float32(self.boundingBox)
        
        modelOffsetPos = br.pos()
        br.write_uint64(0)
        br.write_uint32(0)
        
        entriesOffsetPos = br.pos()
        br.write_uint32(0)
        
        materialsOffsetPos = br.pos()
        br.write_uint32(0)
        
        paramsOffsetPos = br.pos()
        br.write_uint32(0)
        
        namesOffsetPos = br.pos()
        br.write_uint64(0)
        
        submeshOffsetPos = br.pos()
        br.write_uint32(0)
        
        trianglesOffsetPos = br.pos()
        br.write_uint32(0)
        
        texturesOffsetPos = br.pos()
        br.write_uint64(0)
        
        matTexOffsetPos = br.pos()
        br.write_uint32(0)
        
        verticesOffsetPos = br.pos()
        br.write_uint32(0)
        
        unkSecOffsetPos = br.pos()
        br.write_uint64(0)
        
        br.write_uint32(len(self.models))
        br.write_uint64(0)
        
        entriesCountPos = br.pos()
        br.write_uint32(0)
        
        materialCountPos = br.pos()
        br.write_uint32(0)
        
        paramCountPos = br.pos()
        br.write_uint32(0)
        
        br.write_uint64(0) #I don't know what this value does but it can be 0
        
        submeshCountPos = br.pos()
        br.write_uint32(0)
        
        trianglesCountPos = br.pos()
        br.write_uint32(0)
        
        textureCountPos = br.pos()
        br.write_uint64(0)
        
        matTexCountPos = br.pos()
        br.write_uint32(0)
        
        vertexCountPos = br.pos()
        br.write_uint32(0)
        
        if len(self.bones):
            idxTblOffsetPos = br.pos()
            br.write_uint32(0)
            
            indicesOffsetPos = br.pos()
            br.write_uint32(0)
            
            matricesOffsetPos = br.pos()
            br.write_uint32(0)
            
            hierarchyOffsetPos = br.pos()
            br.write_uint32(0)
            
            idxTblCountPos = br.pos()
            br.write_uint32(0)
            
            indicesCountPos = br.pos()
            br.write_uint32(0)
            
            boneCountPos = br.pos()
            br.write_uint32(0)
            
            boneCount2Pos = br.pos()
            br.write_uint32(0)
            
            extraInfoOffsetPos = br.pos()
            br.write_uint32(0)
            
            hierarchyOffset2Pos = br.pos()
            br.write_uint32(0)
        
        br.align(16)
            
        #we have to write the rest of the file before writing the rest of the header
        matBuffer = BinaryReader()
        paramBuffer = BinaryReader()
        matTexBuffer = BinaryReader()
        texBuffer = BinaryReader()
        
        matTexList = []
        paramsList = []
        
        # write materials
        matBuffer.write_struct(self.materials, matTexList, paramsList)
        print("Materials Written")
        
        #write params
        for param in paramsList:
            paramBuffer.pad(4)
            paramBuffer.write_float32(param)
        print("Params Written")
        
        #write material textures
        matTexBuffer.write_struct(matTexList, self.textures)
        print("Material Textures Written")
        
        #write texture info
        texBuffer.write_struct(self.textures)
        print("Textures Written")
        
        #write the submeshes
        trianglesList = []
        verticesList = []
        
        submeshBuffer = BinaryReader()
        submeshBuffer.write_struct(self.submeshes, trianglesList, verticesList)
        print("submeshes Written")
        
        #write the vertex buffer
        vertexBuffer = BinaryReader()
        
        for vertex in verticesList:
            vertex: TMD2Vertex
            if self.modelFlags & 0x2:
                vertexBuffer.write_float32(vertex.position)
                
            if self.modelFlags & 0x400:
                vertexBuffer.write_uint8([int(w * 255) for w in vertex.boneWeights])
                vertexBuffer.write_uint8(vertex.boneIDs)
                
            if self.modelFlags & 0x8000:
                vertexBuffer.write_uint8([int(w * 255) for w in vertex.boneWeights2])
                vertexBuffer.write_uint8(vertex.boneIDs2)
                
            if self.modelFlags & 0x4:
                vertexBuffer.write_uint8([int((f * 0.5 + 0.5) * 255 + 0.5) for f in vertex.normal])
                vertexBuffer.write_uint8(255)
                
            if self.modelFlags & 0x8:
                vertexBuffer.write_uint8([int((f * 0.5 + 0.5) * 255 + 0.5) for f in vertex.tangent])
                vertexBuffer.write_uint8(0)
                vertexBuffer.write_uint8([int((f * 0.5 + 0.5) * 255 + 0.5) for f in vertex.binormal])
                vertexBuffer.write_uint8(0)
                
            if self.modelFlags & 0x80:
                vertexBuffer.write_uint8([int(c * 255) for c in vertex.color])
            
            if self.modelFlags & 0x100:
                vertexBuffer.write_uint8([int((f * 0.5 + 0.5) * 255 + 0.5) for f in vertex.normal2])
                vertexBuffer.write_uint8(255)
            
            if self.modelFlags & 0x200:
                vertexBuffer.write_uint8([int(c * 255) for c in vertex.color2])
            
            if self.modelFlags & 0x10:
                vertexBuffer.write_float32(vertex.uv)
            
            if self.modelFlags & 0x20:
                vertexBuffer.write_float32(vertex.uv2)
            
            if self.modelFlags & 0x40:
                vertexBuffer.write_float32(vertex.uv3)
        
        print("vertex buffer written")
        
        #write triangles buffer
        triBuffer = BinaryReader()
        
        if self.modelFlags & 0x800:
            for triangle in trianglesList:
                triBuffer.write_uint32(triangle)
        else:
            for triangle in trianglesList:
                triBuffer.write_uint16(triangle)

        print("triangles buffer written")
        
        #write models
        modelBuffer = BinaryReader()
        namesBuffer = BinaryReader(encoding="cp932")        
        
        indexTablesList = []
        submeshEntries = []
        modelBuffer.write_struct(self.models, indexTablesList, submeshEntries, self.submeshes, self.materials, namesBuffer)
        
        #write submesh entries
        entryBuffer = BinaryReader()
        entryBuffer.write_struct(submeshEntries)
        
        #write bone buffers
        boneMatrixBuffer = BinaryReader()
        boneHierarchyBuffer = BinaryReader()
        boneExtraInfoBuffer = BinaryReader()
        unkBoneBuffer = BinaryReader()
        
        for bone in self.bones:
            boneMatrixBuffer.write_float32([value for row in bone.matrix for value in row])
            boneHierarchyBuffer.write_struct(bone, namesBuffer, self.version)
            boneExtraInfoBuffer.write_int16(bone.extra)
        
        for unkInfo in self.unkBoneInfo:
            unkBoneBuffer.write_float32(unkInfo)
        
        print("Bone Data Written")
        
        #write index tables
        indexTblBuffer = BinaryReader()
        indicesBuffer = BinaryReader()
        
        indicesList = []
        indexTblBuffer.write_struct(indexTablesList, indicesList)
        indicesBuffer.write_uint32(indicesList)
        
        print("Indices Written")
        
        unkSecBuffer = BinaryReader()
        unkSecBuffer.write_struct(self.unkSections)
        
        print("Unknown section written")
        
        # Write all buffers and record their offsets
        def write_and_get_offset(buffer: BinaryReader, alignment=16):
            offset = br.pos()
            br.write_bytes(bytes(buffer.buffer()))
            br.align(alignment)
            return offset


        modelOffset = write_and_get_offset(modelBuffer)
        entriesOffset = write_and_get_offset(entryBuffer)
        materialsOffset = write_and_get_offset(matBuffer)
        paramsOffset = write_and_get_offset(paramBuffer)
        submeshOffset = write_and_get_offset(submeshBuffer)
        trianglesOffset = write_and_get_offset(triBuffer)
        texturesOffset = write_and_get_offset(texBuffer)
        matTexOffset = write_and_get_offset(matTexBuffer)
        verticesOffset = write_and_get_offset(vertexBuffer)
        unkSecOffset = write_and_get_offset(unkSecBuffer)

        # Optional bone section
        if len(self.bones):
            if len(self.bones) >= 255:
                indexTablesOffset = write_and_get_offset(indexTblBuffer)
                tableIndicesOffset = write_and_get_offset(indicesBuffer)
            else:
                indexTablesOffset = tableIndicesOffset = 0
                indicesList = []
                indexTablesList = []
            boneMatrixOffset = write_and_get_offset(boneMatrixBuffer)
            boneHierarchyOffset = write_and_get_offset(boneHierarchyBuffer)
            extraBoneInfoOffset = write_and_get_offset(boneExtraInfoBuffer)
            unkBoneOffset = write_and_get_offset(unkBoneBuffer)
        
        namesOffset = write_and_get_offset(namesBuffer)

        # Patch header
        def patch_u32(pos, val):
            here = br.pos()
            br.seek(pos)
            br.write_uint32(val)
            br.seek(here)

        def patch_u64(pos, val):
            here = br.pos()
            br.seek(pos)
            br.write_uint64(val)
            br.seek(here)

        patch_u64(modelOffsetPos, modelOffset)
        patch_u32(entriesOffsetPos, entriesOffset)
        patch_u32(materialsOffsetPos, materialsOffset)
        patch_u32(paramsOffsetPos, paramsOffset)
        patch_u64(namesOffsetPos, namesOffset)
        patch_u32(submeshOffsetPos, submeshOffset)
        patch_u32(trianglesOffsetPos, trianglesOffset)
        patch_u64(texturesOffsetPos, texturesOffset)
        patch_u32(matTexOffsetPos, matTexOffset)
        patch_u32(verticesOffsetPos, verticesOffset)
        patch_u64(unkSecOffsetPos, unkSecOffset)

        patch_u32(entriesCountPos, len(submeshEntries))
        patch_u32(materialCountPos, len(self.materials))
        patch_u32(paramCountPos, len(paramsList))
        patch_u32(submeshCountPos, len(self.submeshes))
        patch_u32(trianglesCountPos, len(trianglesList))
        patch_u64(textureCountPos, len(self.textures))
        patch_u32(matTexCountPos, len(matTexList))
        patch_u32(vertexCountPos, len(verticesList))

        if len(self.bones):
            patch_u32(idxTblOffsetPos, indexTablesOffset)
            patch_u32(indicesOffsetPos, tableIndicesOffset)
            patch_u32(matricesOffsetPos, boneMatrixOffset)
            patch_u32(hierarchyOffsetPos, boneHierarchyOffset)
            patch_u32(idxTblCountPos, len(indexTablesList))
            patch_u32(indicesCountPos, len(indicesList))
            patch_u32(boneCountPos, len(self.bones))
            patch_u32(boneCount2Pos, len(self.bones))
            patch_u32(extraInfoOffsetPos, extraBoneInfoOffset)
            patch_u32(hierarchyOffset2Pos, unkBoneOffset)
    
    
        print("done")
        
            
        

class TMD2Unk(BrStruct):
    def __init__(self) -> None:
        self.values = [0] * 24
    
    def __br_read__(self, br: 'BinaryReader') -> None:
        self.values = br.read_float32(24)
    
    def __br_write__(self, br, *args):
        br.write_float32(self.values)


class TMD2Model(BrStruct):
    def __init__(self) -> None:
        self.boundingBox = [0,0,0,0,0,0]
        self.meshes = []
        self.materials = []
        self.name = ""
        self.hash = 0
    
    def __br_read__(self, br: 'BinaryReader', indexTables, entries, submeshes, materials, namesOffset) -> None:
        self.boundingBox = br.read_float32(6)
        self.entriesCount = br.read_uint16()
        self.unk0 =  br.read_uint16()
        self.entriesStart = br.read_uint32()
        self.nameOffset = br.read_int32()
        self.hash =  br.read_uint32()
        self.unk1 =  br.read_uint32()
        
        if namesOffset > 0 and self.nameOffset != -1:
            self.name = br.read_str_at_offset(namesOffset + self.nameOffset)
        else:
            self.name = f"{str(self.hash)}"
        
        materialIdx = -1
        indexTable = indexTables[0] if indexTables else TMD2IndexTable()
        
        for entry in entries[self.entriesStart: self.entriesStart + self.entriesCount]:
            if entry.type == 96:
                self.materials.append(materials[entry.index])
                materialIdx += 1
            
            elif entry.type == 64:
                indexTable = indexTables[entry.index]
            elif entry.type == 48:
                mesh = submeshes[entry.index]
                mesh.materialIndex = materialIdx
                mesh.material = materials[materialIdx]
                mesh.indexTable = indexTable.indices
                self.meshes.append(mesh)
            
    
    def __br_write__(self, br: 'BinaryReader', indexTables, entries, submeshes, materials, namesBuffer) -> None:
        br.write_float32(self.boundingBox)
        #br.write_uint16(len(self.meshes))
        
        #write meshes
        materialIdx = -1
        submeshIdx = len(submeshes)
        
        localEntries = []
        
        for mesh in self.meshes:
            mesh: TMD2Submesh
            materialIndex = materials.index(mesh.material)
            if materialIdx != materialIndex:
                #update the material index
                materialIdx = materialIndex
                
                #create a material entry
                matEntry = TMD2SubmeshEntry()
                matEntry.type = 96
                matEntry.index = materialIdx
                localEntries.append(matEntry)
            
            if mesh.indexTable:
                
                idxTbl = TMD2IndexTable()
                idxTbl.indices = mesh.indexTable
                
                #create index table entry
                idxTblEntry = TMD2SubmeshEntry()
                idxTblEntry.type = 64
                idxTblEntry.index = len(indexTables)
                
                indexTables.append(idxTbl)                
                localEntries.append(idxTblEntry)
                
                
            #create mesh entry
            meshEntry = TMD2SubmeshEntry()
            meshEntry.index = submeshIdx
            meshEntry.type = 48
            
            localEntries.append(meshEntry)
        
        #when done create end Entry
        endEntry = TMD2SubmeshEntry()
        endEntry.index = 0
        endEntry.type = 16
        localEntries.append(endEntry)
                
        
        #write the amount of local entries then extend the entries list
        br.write_uint16(len(localEntries))
        br.write_uint16(self.unk0)
        br.write_uint32(len(entries))
        entries.extend(localEntries)
        
        
        if int(self.name) != self.hash:
            br.write_int32(namesBuffer.size())
            namesBuffer.write_str(self.name)
        else:
            br.write_int32(-1)
        
        br.write_uint32(self.hash)
        br.write_uint32(0)
        
        


class TMD2Submesh(BrStruct):
    def __init__(self):
        self.material = None
        self.indexTable = []
        self.triangles = []
        self.vertices = []
        self.materialIndex = 0
        self.trianglesCount = 0
        self.trianglesStart = 0
    
    
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
    
    def __br_write__(self, br, trianglesList, verticesList):
        br.write_uint32(len(self.triangles))  # Write triangle count

        # Store the starting triangle index
        t_startIndex = len(trianglesList)
        br.write_uint32(t_startIndex)

        # Map from local vertex index (within this submesh) to global index (in verticesList)
        local_to_global = {}

        for i, v in enumerate(self.vertices):
            global_idx = len(verticesList)
            verticesList.append(v)
            local_to_global[i] = global_idx

        # Remap triangle vertex indices using local_to_global and append to the global triangle list
        for tri in self.triangles:
            global_tri = [local_to_global[vi] for vi in tri]
            trianglesList.append(global_tri)
        
        

class TMD2SubmeshEntry(BrStruct):
    def __init__(self):
        self.type = None
        self.index = -1
    
    def __br_read__(self, br, *args):
        self.index = br.read_uint8()
        self.type = br.read_uint8()
    
    def __br_write__(self, br, *args):
        br.write_int8(self.index)
        br.write_int8(self.type)


class TMD2Texture(BrStruct):
    def __init__(self) -> None:
        self.hash = 0
        self.index = 0
        self.width = 0
        self.height = 0
        self.format = 21074
        self.data = b''
    
    def __br_read__(self, br: 'BinaryReader', *args) -> None:
        self.hash = br.read_uint32()
        self.index = br.read_uint16()
        self.width = br.read_uint16()
        self.height = br.read_uint16()
        self.format = br.read_uint16()
    
    
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        br.write_uint32(self.hash)
        br.write_uint16(self.index)
        br.write_uint16(self.width)
        br.write_uint16(self.height)
        br.write_uint16(self.format)
        


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
        self.unk1 = br.read_int16()
        self.unk2 = br.read_int16()
        self.slot = br.read_int16() >> 8
    
    def __br_write__(self, br, textures):
        br.write_uint32(self.texture.hash)
        br.write_uint16(textures.index(self.texture))
        br.write_int16(self.unk1)
        br.write_int16(self.unk2)
        br.write_int8(0)
        br.write_int8(self.slot)

class TMD2Material(BrStruct):
    def __init__(self) -> None:
        self.hash = 0
        self.name = ''
        self.shaderID = ""
        self.textures = []
        self.shaderParams = []
        self.unk = -1

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
        params.extend(self.shaderParams)
        br.write_int16(self.unk)

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
    
    
    def __br_write__(self, br, namesBuffer: BinaryReader, version):
        br.write_uint32(self.hash)
        br.write_float32(self.parentIndex)
        br.write_int32(self.parentIndex)
        br.write_uint16(0)
        
        if version >= 0x209:
            br.write_uint16(namesBuffer.size())
            namesBuffer.write_str(self.name)
        else:
            br.write_int16(-1)


class TMD2IndexTable(BrStruct):
    def __init__(self):
        self.indices = []
        self.indicesCount = 0
    
    def __br_read__(self, br, tableIndices):
        
        self.indicesCount = br.read_uint32()
        self.startIndex = br.read_uint32()
        
        self.indices = tableIndices[self.startIndex: self.startIndex + self.indicesCount]

    def __br_write__(self, br, indicesList):
        br.write_uint32(len(self.indices))
        br.write_uint32(len(indicesList))
        indicesList.extend(self.indices)