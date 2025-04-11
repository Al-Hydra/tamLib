import os
import zlib
from .utils.PyBinaryReader.binary_reader import *


class PZZEFile(BrStruct):
    def __init__(self):
        self.magic = 'PZZE'
        self.dataOffset = 24
        self.fileFormat = ''
        self.decompressedSize = 0
        self.compressedData = b''
        self.decompressedData = b''
    
    def __br_read__(self, br: BinaryReader, *args) -> None:
        self.magic = br.read_str(4)
        if self.magic != 'PZZE':
            print(f"{self.magic} is not a PZZE file.")
            return None
        
        self.fileFormat = br.read_str(4)
        self.decompressedSize = br.read_uint64()
        self.dataOffset = br.read_uint64()
        self.compressedData = br.buffer()[self.dataOffset:]
    
    def __br_write__(self, br: BinaryReader, fileFormat = "tmd2") -> None:
        br.write_str(self.magic)
        br.write_str(fileFormat)
        br.write_uint64(len(self.decompressedData))
        br.write_uint64(24)
        br.write_bytes(self.compress())
        
    def decompress(self):
        if self.compressedData:
            try:
                self.decompressedData = zlib.decompress(self.compressedData)
            except zlib.error:
                print(f"Decompression failed.")
                return None
        else:
            print(f"No compressed data found.")
            return None
        return self.decompressedData
    
    def compress(self):
        if self.decompressedData:
            try:
                self.compressedData = zlib.compress(self.decompressedData)
            except zlib.error:
                print(f"Compression failed.")
                return None
        else:
            print(f"No decompressed data found.")
            return None
        return self.compressedData


def readPZZE(path):
    with open(path, "rb") as f:
        br = BinaryReader(f.read())
        pzze = br.read_struct(PZZEFile)
        return pzze


if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.abspath(__file__))
    outputDir = os.path.join(fileDir, "output")
    os.makedirs(outputDir, exist_ok=True)
    
    path =r"G:\Dev\io_tmd_tmo\pl002_hair00_00.tmd2"
    filename = os.path.basename(path)

    pzze = readPZZE(path)
    output = pzze.decompress()
    if output:
        baseName = os.path.splitext(filename)[0]
        fileOutputDir = os.path.join(outputDir, baseName)
        os.makedirs(fileOutputDir, exist_ok=True)
        
        #save the decompressed file
        outputPath = os.path.join(fileDir, baseName + "_decompressed" + os.path.splitext(filename)[1])
        with open(outputPath, "wb") as f:
            f.write(output)
        
        '''
        #compress test
        new_pzze = PZZEFile()
        new_pzze.decompressedData = output
        new_pzze.fileFormat = "TEST"
        
        new_br = BinaryReader()
        new_br.write_struct(new_pzze, pzze.fileFormat)
        
        #new file name
        new_file_name = os.path.splitext(filename)[0] + "_compressed" + os.path.splitext(filename)[1]
        new_file_path = os.path.join(fileDir, new_file_name)
        with open(new_file_path, "wb") as f:
            f.write(new_br.buffer())'''