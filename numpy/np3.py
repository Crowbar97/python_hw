import numpy as np

def myprint(msg, obj):
    print("{}:\n{}".format(msg, obj))

print("- zeros insertion:")
a = [1, 2, 3, 4, 5]
myprint("a", a)
b = [a[0]]
for i in range(1, len(a)):
    b.extend([0, 0, 0, a[i]])
myprint("b", b)

print("\n- row swapping:")
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
myprint("initial", m)
m[[0, 2]] = m[[2, 0]]
myprint("swapped", m)

print("\n- unique segments:")
ts = [[(1, 2), (3, 4), (5, 6)],
      [(4, 8), (3, 5), (1, 2)],
      [(4, 4), (5, 6), (5, 7)]]
ts = np.array(ts)
myprint("initial", ts)
myprint("unique", np.unique(ts.reshape(ts.shape[0] * ts.shape[1], 2), axis=0))

print("\n- bincount nonsense:")
c = [2, 2, 1, 0, 1]
myprint("c", c)
a = [0, 0, 1, 1, 2, 4]
myprint("bincount a", np.bincount(a))

print("\n- 1D convolution:")
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
myprint("initial", x)
h = 3
conv = np.convolve(x, np.ones((h,)) / h, mode='valid')
myprint("conv", conv)

print("\n- shifted matrix:")
def get_fwd(arr, beg, end):
    r = range(beg, end)
    ids = list(map(lambda ind: ind % len(arr), r))
    return arr[ids]

arr = np.arange(8)
myprint("initial", arr)
m = []
for i in range(0, -4, -1):
    m.append(get_fwd(arr, i, i + 3))
myprint("shifted", np.array(m))

print("\n- inversion:")
arr = np.array([True, False, True])
myprint("source", arr)
np.logical_not(arr, out=arr)
myprint("inversed:", arr)

arr = np.array([0.1, -0.2, 0.1])
myprint("\nsource", arr)
np.negative(arr, out=arr)
myprint("inversed:", arr)

print("\n- rank calc:")
m = np.eye(4)
myprint("matrix", m)
myprint("rank", np.linalg.matrix_rank(m))

print("\n- the most frequent value:")
arr = np.array([1, 2, 1, 2, 2, 3, 2])
myprint("source", arr)
myprint("most freq", np.bincount(arr).argmax())

print("\n- adjacent blocks:")
m = np.array(range(100)).reshape((10, 10))
myprint("source:", m)

h = 3
for i in range(0, m.shape[0] - h, h):
    for j in range(0, m.shape[1] - h, h):
        myprint("(%d, %d)" % (i, j), m[i : i + h, j : j + h])

print("\n- tensor product:")
p, n = 3, 4
ms = np.ones((p, n, n))
vs = np.ones((p, n, 1))
s = np.tensordot(ms, vs, axes=[[0, 2], [0, 1]])
myprint("sum", s)

print("\n- 2D convolution:")
m = np.ones((16,16))
myprint("source", m)
k = 4
s = np.add.reduceat(
        np.add.reduceat(m, np.arange(0, m.shape[0], k), axis=0),
        np.arange(0, m.shape[1], k), axis=1)
myprint("sum", s)

print("\n- n max values:")
arr = np.arange(10)
np.random.shuffle(arr)
myprint("source", arr)
n = 5
myprint("top %d" % n, np.sort(arr)[-n:])

print("\n- cartesian:")
# def cart(arrs):
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    # print("shape:", shape)

    ix = np.indices(shape, dtype=int)
    # print("ix:", ix)
    ix = ix.reshape(len(arrays), -1).T
    # print("ix rs:", ix)

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

arr = [[1, 2, 3], [4, 5], [6, 7]]
myprint("source", arr)
myprint("cart", cartesian(arr))

print("\n- element containing")
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
myprint("a", a)
b = np.array([[6, 4],
              [6, 5]])
myprint("b", b)
b_u = np.unique(b)
print("Next rows of \"a\" contain \"b\":")
for i, row in enumerate(a):
    if np.isin(row, b_u).all():
        print(i)

print("\n- unequal rows searching:")
z = np.random.randint(0, 5, (10, 3))
z[2] = [3, 3, 3]
z[5] = [8, 8, 8]
myprint("source", z)
u = z[z.max(axis=1) != z.min(axis=1), :]
myprint("unequal rows", u)

print("\n- int to bin:")
vi = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
myprint("source", vi)
myprint("bin", np.unpackbits(vi[:, np.newaxis], axis=1))

print("\n- unique rows:")
z = np.array([[1, 2, 3],
              [4, 5, 6],
              [1, 2, 3]])
myprint("source", z)
myprint("unique", np.unique(z, axis=0))

print("\n- Ein notation:")
a = np.random.randint(0, 10, 5)
b = np.random.randint(0, 10, 5)
myprint("a", a)
myprint("b", b)

# np.sum(a)
myprint("sum", np.einsum('i->', a))
# a * b
myprint("mul", np.einsum('i,i->i', a, b))
# np.inner(a, b)
myprint("inner", np.einsum('i,i', a, b))
# np.outer(a, b)
myprint("outer", np.einsum('i,j->ij', a, b))

