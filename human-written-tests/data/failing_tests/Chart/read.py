tot = 0
for i in range(1,27):
    with open(str(i), "r") as f:
        tot += len(f.readlines())
print(tot)