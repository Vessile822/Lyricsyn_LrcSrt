import struct
import sys

if sys.version_info[:2] not in ((3, 11), (3, 12)) or struct.calcsize("P") != 8:
    raise SystemExit(1)
print(sys.executable)
