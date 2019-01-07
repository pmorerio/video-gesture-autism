import numpy as np

print 'If not working, run in ipython from /old/xxxx model dir as "% run ../../util/read_history.py"'

model = np.load('model.npz')
history = model['history_acc']

history = np.array(history)

#print 'Max valid acc', history[:,0].max()
#print 'Max test acc', history[:,1].max()

print 'Train accuracy'
for item in history[:,0]:
	print item
print 'Test accuracy'
for item in history[:,2]:
	print item
