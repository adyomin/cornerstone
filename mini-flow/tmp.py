stride = 2
patch = 3
side = 28
counter = 0

for i in range(patch, side + 1, stride):
    counter += 1
    print (i)

print('Counter at the ent = {0}'.format(counter))