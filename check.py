import csv
import os
import math


def chequear_claves(A, B):
	for k in A:
		if k not in B:
			return k
	return None
	
## leo la solucion en S
f_in = open('ml-latest-small/data/ratings_test.csv', 'rt')
csvreader = csv.DictReader(f_in)

S = dict()
for row in csvreader:
	S[ (int(row["userId"]),int(row["movieId"])) ] = float(row["rating"])

f_in.close()

E = []

for f in os.listdir("ml-latest-small"):
	if not f.endswith(".csv"):
		continue

	fn = os.path.join("ml-latest-small", f)

	# leo respuesta
	f_in = open(fn, 'rt')
	csvreader = csv.DictReader(f_in)

	R = dict()
	for row in csvreader:
		R[ (int(row["userId"]),int(row["movieId"])) ] = float(row["rating"])
	f_in.close()

	# chequeo claves
	k = chequear_claves(S, R)	
	if k:
		print("ERROR: La clave", k, "no está en", f)
		continue
		
	
	error = math.sqrt(sum([ (S[k] - R[k])**2 for k in S]) / len(S.keys()))
	E.append((f.replace('.csv', ''), error))

print("SOLUCION   " + '\t' + "RMSE")
for f, e in sorted(E, key=lambda x: x[1]):
	print(f + '\t' + str(e))

