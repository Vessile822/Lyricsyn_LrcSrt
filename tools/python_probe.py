import struct
import sys

if sys.version_info[:2] < (3, 11) or struct.calcsize("P") != 8:
    raise SystemExit(1)
print(sys.executable)
