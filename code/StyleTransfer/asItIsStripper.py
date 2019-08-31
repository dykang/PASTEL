import sys

inFileName=sys.argv[1]

outFileName=inFileName+".asItIs"
outFile=open(outFileName,"w")

headerLength=int(sys.argv[2])

for line in open(inFileName).readlines():
    words=line.split()
    words=words[headerLength:]
    outLine=" ".join(words)+"\n"
    outFile.write(outLine)

outFile.close()


