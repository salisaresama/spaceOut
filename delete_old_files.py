import os

def main():
	x = os.listdir("/data/tmp/")
	y = []
	for i in x:
		if i[0] != ".":
			n = int(i[0:8])
			if n < 20200414:
				y.append(i)

	for i in y:
		os.remove("/data/tmp/" + i)

main()
