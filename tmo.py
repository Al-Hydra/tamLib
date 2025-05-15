from .utils.PyBinaryReader.binary_reader import *
import struct

class TMO(BrStruct):
    def __init__(self):
        self.name = ""
        self.magic = b""
        self.offset = 0
        self.flag1 = 0
        self.scale = 0.0
        self.framesOffset = 0
        self.unk = 0
        self.boneHashesOffset = 0
        self.pointerCount = 0
        self.pointerOffset = 0
        self.keyframeCount = 0
        self.keyframeOffset = 0
        self.frameCount = 0
        self.boneCount = 0
        self.hashes = []
        self.offsets = []
        self.keyframes = []
    
    def __br_read__(self, br: BinaryReader) -> None:
        self.magic = br.read_str(4)
        self.offset = br.read_uint32()
        self.flag1 = br.read_uint32()
        self.scale = br.read_float32()
        
        '''self.flag2 = br.read_uint64()
        self.framesOffset = br.read_uint64()
        br.seek(16, 1)
        self.unk = br.read_uint64()
        self.boneHashesOffset = br.read_uint64()
        self.pointerCount = br.read_uint64()
        self.pointerOffset = br.read_uint64()
        self.keyframeCount = br.read_uint64()
        self.keyframeOffset = br.read_uint64()'''
        
        self.flag2 = br.read_uint32()
        self.framesOffset = br.read_uint32()
        #br.seek(16, 1)
        br.align_pos(16)
        self.unk = br.read_uint64()
        self.boneHashesOffset = br.read_uint32()
        self.pointerCount = br.read_uint32()
        self.pointerOffset = br.read_uint32()
        self.keyframeCount = br.read_uint32()
        self.keyframeOffset = br.read_uint32()

        # Read frame and bone counts
        br.seek(self.framesOffset, Whence.BEGIN)
        self.frameCount = br.read_uint16()
        self.boneCount = br.read_uint16()

        # Read bone hashes
        br.seek(self.boneHashesOffset, Whence.BEGIN)
        for i in range(self.boneCount):
            self.hashes.append(br.read_uint32())

        # Read offsets
        br.seek(self.pointerOffset, Whence.BEGIN)
        if self.flag1 & 1:
            for i in range(self.pointerCount):
                offset_data = {
                    "startFrame": br.read_uint32(),
                    "frameCount": br.read_uint32(),
                    "lastFrame": br.read_uint32()
                }
                self.offsets.append(offset_data)
        else:
            for i in range(self.pointerCount):
                offset_data = {
                    "startFrame": br.read_uint64(),
                    "frameCount": br.read_uint32(),
                    "lastFrame": br.read_uint32()
                }
                self.offsets.append(offset_data)


        indexDict = {0: "px",
                 1: "py",
                 2: "pz",
                 3: "rx",
                 4: "ry",
                 5: "rz",
                 6: "sx",
                 7: "sy",
                 8: "sz"}
        
        boneFramesDict = {"px": [],
                          "py": [],
                          "pz": [],
                          "rx": [],
                          "ry": [],
                          "rz": [],
                          "sx": [],
                          "sy": [],
                          "sz": []}

        # Read keyframes
        br.seek(self.keyframeOffset, Whence.BEGIN)
        if self.flag1 & 1:
            for i in range(self.keyframeCount):
                frame = br.read_uint16()
                lastFrame = br.read_uint16()
                value = br.read_float32()
                self.keyframes.append({frame: value})
        else:
            for i in range(self.keyframeCount):
                frame = br.read_uint16()
                value = br.read_uint16()
                self.keyframes.append({frame: value})

        

if __name__ == "__main__":
    # Example usage
    tmo_path = r"C:\Users\Hydra\Desktop\anim.tmo"
    
    with open(tmo_path, "rb") as f:
        data = f.read()
        br = BinaryReader(data)
        tmo = TMO()
        tmo.__br_read__(br)
        print(tmo.hashes)
        print(tmo.offsets)
        print(tmo.keyframes)