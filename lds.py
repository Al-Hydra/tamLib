from .utils.PyBinaryReader.binary_reader import *


class LDS(BrStruct):
    
    def __init__(self):
        self.name = ""
        self.textureCount = 0
        self.textures = []
        self.unk = 0
    
    def __br_read__(self, br: BinaryReader, file_name = "") -> None:
        self.name = file_name
        self.unk = br.read_uint32()
        self.textureCount = br.read_uint32()
        self.fileSize = br.read_uint32()
        
        offsets = []
        
        for i in range(self.textureCount):
            offsets.append(br.read_uint32())
        
        
        pos = br.pos()
        for i in range(self.textureCount):
            br.seek(offsets[i] + pos, Whence.BEGIN)
            
            if i != self.textureCount - 1:
                textureSize = offsets[i + 1] - offsets[i]
                self.textures.append(br.read_bytes(textureSize))
                print(f"Texture {i} size: {textureSize}")
            else:
                textureSize = br.size() - offsets[i] - pos
                print(f"Texture {i} size: {textureSize}")
                self.textures.append(br.read_bytes(textureSize))
        
        
    def __br_write__(self, br: BinaryReader, *args) -> None:
        
        br.write_uint32(self.unk)
        br.write_uint32(len(self.textures))
        
        sizePos = br.pos()
        br.write_uint32(0)
        
        texturesBuffer = BinaryReader()
        for i in range(len(self.textures)):
            texturesBuffer.write_bytes(self.textures[i])
            br.write_uint32(texturesBuffer.size())
        br.write_bytes(texturesBuffer.buffer())


if __name__ == "__main__":
    # Example usage
    lds_path = r"G:\Dev\BlenderTMD2\tamLib\bg_adv_001tex0_decompressed.lds"
    
    with open(lds_path, "rb") as f:
        data = f.read()
    
    br = BinaryReader(data, endianness=Endian.LITTLE)
    lds = br.read_struct(LDS)

    #dump
    path = r"G:\Dev\BlenderTMD2\tamLib\output"
    
    for i, texture in enumerate(lds.textures):
        with open(f"{path}/texture_{i}.dds", "wb") as f:
            f.write(texture)