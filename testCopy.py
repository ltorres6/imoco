# testCopy

import subprocess

subject = "103-005"
visit = "20160114"
contrast="pre"
# Copy Raw Data
subprocess.call(["/export/home/ltorres/projects/xdgrasp/copyData.sh", "lat205", subject, visit, contrast])

contrast="post"
subprocess.call(["/export/home/ltorres/projects/xdgrasp/copyData.sh", "lat205", subject, visit, contrast])
