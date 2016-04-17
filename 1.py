f1 = open('2.log', 'w')
f2 = open('3.log', 'w')
with open('1.log', 'r') as f:
    lines = f.readlines()
    for idx in xrange(len(lines)):
        if idx % 2 == 0:
            f1.write(lines[idx])
        else:
            f2.write(lines[idx])
      
    
