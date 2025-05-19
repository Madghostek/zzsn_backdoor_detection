from pathlib import Path
import numpy as np



all_series = []
with Path("MM-BD/MM_BD_outputs.txt").open("r") as f:
    serie = []    
    for line in f.readlines():
        if line == "\n":
            all_series.append(serie)
            serie = []
        else:
            serie.append(line.rstrip())

correct_count = 0
for serie in all_series:
    values = [float(l.split(" ")[1]) for l in serie[:4]]
    name = serie[6]
    amax = serie[-1][-1]
    try:
        amax = int(amax)
    except:
        amax = -1
    correct = int(name[-5])+1 if name[-6]=="p" else -1
    correct_count+=correct==amax
    print(name,amax,correct,correct==amax)

print(correct_count/len(all_series))
