def Action1(q):
	print 'Action 1', q
def Action2(p, q):
	print 'Action2', p, q
def Action3(p, r, q):
	print 'Action3', p, r, q

def Perform(f):
    f(4)

from functools import partial

Perform(Action1)
Perform(partial(Action2, 11))
Perform(partial(Action3, 22, 33))