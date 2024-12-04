import numpy as np
x = 1

class A:
    def __init__(self, x):
        self.x = x

def f(a):
    if isinstance(a, A):
        a.x = a.x + 1
        print("Value of object in f: ", a.x)
        return
    a = a + 1
    print("Value of a in f: ", a)
    return

f(x)
print("Value of a in main:", x) # Python uses pass by value

y = A(1)
f(y)
print("Value of object in main:", y.x) # Objects are passed by reference

idx = (
    [0, 0, 0, 1, 1, 1],
    [0, 1, 2, 0, 1, 2]
)

arr = np.array([[1, 2, 3], [4, 5, 6]])

oneDarr = np.array([11, 22, 33, 44, 55, 66])

arr[idx[0], idx[1]] = oneDarr

print(arr)
