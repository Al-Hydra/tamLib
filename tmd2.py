from utils.PyBinaryReader.binary_reader import *
import numpy as np
class TMD2(BrStruct):

    def __init__(self) -> None:
        self.magic = 'tmd0'
        
    def __br_read__(self, br: 'BinaryReader', *args) -> None:

        self.magic = br.read_str(4)
        if self.magic != 'tmd0':
            raise ValueError(f"Invalid magic: {self.magic}")
        
        unk0 = br.read_uint16()
        self.vertexFlag1 = br.read_uint8()
        self.vertexFlag2 = br.read_uint8()
        unk1 = br.read_uint32()
        unk2 = br.read_uint16()
        self.frameCount = br.read_int16()
        self.boundingBox = br.read_float32(6)
        self.modelsOffset = br.read_uint64()
        unk3 = br.read_uint32()
        self.subMeshOffset = br.read_uint32()
        self.materialsOffset = br.read_uint32()
        self.shaderParamsOffset = br.read_uint32()
        self.namesOffset = br.read_uint64()
        self.subMeshOffset2 = br.read_uint32()
        self.trianglesOffset = br.read_uint32()
        self.texturesOffset = br.read_uint64()
        self.materialTexturesOffset = br.read_uint32()
        self.verticesOffset = br.read_uint32()
        self.unkOffset = br.read_uint64()
        self.modelCount = br.read_uint32()
        unk4 = br.read_uint32()
        unk5 = br.read_uint32()
        self.subMeshCount = br.read_uint32()
        self.materialCount = br.read_uint32()
        self.shaderParamsCount = br.read_uint32()
        unk6 = br.read_uint32()
        unk7 = br.read_uint32()
        self.subMeshCount2 = br.read_uint32()
        self.trianglesCount = br.read_uint32()
        self.textureCount = br.read_uint32()
        unk8 = br.read_uint32()
        self.materialTextureCount = br.read_uint32()
        self.vertexCount = br.read_uint32()
        self.triangleInfoStart = br.read_uint32()
        unk9 = br.read_uint32()
        self.boneMatrixOffset = br.read_uint32()
        self.boneHierarchyOffset = br.read_uint32()
        unk10 = br.read_uint32()
        unk11 = br.read_uint32()
        self.boneCount = br.read_uint32()
        unk12 = br.read_uint32()
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
        self.shaderParams = [br.read_float64() for _ in range(self.shaderParamsCount)]
        
        #Material Info
        br.seek(self.materialsOffset, Whence.BEGIN)
        self.materials = br.read_struct(TMD2Material, self.materialCount, self.textures, self.shaderParams)

        
        # Define converters for specific attributes
        _attribute_converters = {
            'normal':      lambda v: (v.astype(np.float32) / 127.5) - 1.0,
            'normal2':     lambda v: (v.astype(np.float32) / 127.5) - 1.0,
            'tangent':     lambda v: (v.astype(np.float32) / 127.5) - 1.0,
            'binormal':    lambda v: (v.astype(np.float32) / 127.5) - 1.0,
            'color':       lambda v: v.astype(np.float32) / 255.0,
            'color2':      lambda v: v.astype(np.float32) / 255.0,
            'boneWeights': lambda v: v.astype(np.float32) / 255.0,
            'boneWeights2':lambda v: v.astype(np.float32) / 255.0,
        }

        # Determine vertex attributes based on flags
        vertex_attributes = []
        if self.vertexFlag1 & 2:
            vertex_attributes.append(('position', 'f4', 3))

        if self.vertexFlag2 & 4:
            vertex_attributes.append(('boneWeights', 'u1', 4))
            vertex_attributes.append(('boneIDs', 'u1', 4))

        if self.vertexFlag2 & 2:
            vertex_attributes.append(('boneWeights2', 'u1', 4))
            vertex_attributes.append(('boneIDs2', 'u1', 4))

        if self.vertexFlag1 & 4:
            vertex_attributes.append(('normal', 'u1', 4))

        if self.vertexFlag1 & 8:
            vertex_attributes.append(('tangent', 'u1', 4))
            vertex_attributes.append(('binormal', 'u1', 4))

        if self.vertexFlag1 & 128:
            vertex_attributes.append(('color', 'u1', 4))

        if self.vertexFlag2 & 16:
            vertex_attributes.append(('normal2', 'u1', 4))

        if self.vertexFlag2 & 128:
            vertex_attributes.append(('color2', 'u1', 4))

        if self.vertexFlag1 & 16:
            vertex_attributes.append(('uv', 'f4', 2))
        if self.vertexFlag1 & 32:
            vertex_attributes.append(('uv2', 'f4', 2))
        if self.vertexFlag1 & 64:
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


        print(self.vertices[0].__dict__)


        # Read triangle info
        br.seek(self.triangleInfoStart, Whence.BEGIN)
        if self.vertexFlag2 & 8:
            self.triangles = [br.read_uint32(3) for _ in range(self.trianglesCount // 3)]
        else:
            self.triangles = [br.read_uint16(3) for _ in range(self.trianglesCount // 3)]

        # read skeleton info
        br.seek(self.boneHierarchyOffset, Whence.BEGIN)
        self.bones = br.read_struct(TMD2Bone, self.boneCount, self.namesOffset)
        
        #read the matrix info
        br.seek(self.boneMatrixOffset, Whence.BEGIN)
        for i in range(self.boneCount):
            self.bones[i].matrix = br.read_float32(16)
        
        print(self.bones[0].__dict__)
        
    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        pass


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
        
        self.shaderParamsStartIndex = br.read_uint32()
        self.shaderParamsCount = br.read_uint32()
        self.shaderParams = params[self.shaderParamsStartIndex: self.shaderParamsStartIndex + self.shaderParamsCount]

    def __br_write__(self, br: 'BinaryReader', *args) -> None:
        pass


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
        self.headPosition = [0.0, 0.0, 0.0]
        self.matrix = np.zeros((4, 4), dtype=np.float32)

    def __br_read__(self, br: 'BinaryReader', namesOffset = -1) -> None:
        self.hash = br.read_uint32()
        self.headPosition = br.read_float32(3)
        self.parentIndex = br.read_int32()
        
        self.unk1 = br.read_uint16()
        self.nameOffset = br.read_uint16()
        
        self.name = br.read_str_at_offset(namesOffset + self.nameOffset, encoding='utf-8')


def read_tmd2(file: str) -> TMD2:
    with open(file, 'rb') as f:
        br = BinaryReader(f.read(), Endian.LITTLE)
        tmd2 = TMD2()
        tmd2.__br_read__(br)
        return tmd2


if __name__ == "__main__":
    tmd2 = read_tmd2(r"G:\Dev\io_tmd_tmo\pl000_wep00_00_00_decompressed.tmd2")
    print(tmd2.__dict__)