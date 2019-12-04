import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


print("- matrix multiplication:")
m1 = (np.random.rand(5, 3) * 10).astype("int")
m2 = (np.random.rand(3, 2) * 10).astype("int")
print(m1.dot(m2))

print("\n- sign changing:")
lst = np.arange(0, 10)
print("initial: {}".format(lst))
lst = list(map(lambda x: -x if 3 <= x <= 8 else x, lst))
print("changed : {}".format(lst))

print("\n- [0 - 4] matrix creation:")
m = np.array([range(5) for _ in range(5)])
print("matrix:\n{}".format(m))

print("\n- list generation:")
my_gen = (x ** 2 for x in range(10) if x % 2 == 0)
my_lst = [x for x in my_gen]
print("generated list: {}".format(my_lst))

print("\n- vector creation:")
a, b = 0, 1
h = b / 11
v = np.arange(a + h, b, h)
print("v: {}".format(v))

print("\n- sorting:")
v = np.array([2, 3, 2, 7, 1, 9, 0, 8, 9, 23, 12])
print("initial: {}".format(v))
print("sorted: {}".format(np.sort(v)))

print("\n- equal checking:")
a = [1, 2, 3]
print("a = {}".format(a))
b = [2, 3, 4]
print("b = {}".format(b))
c = [1, 2 ,3]
print("c = {}".format(c))
print("a == b: {}".format(np.array_equal(a, b)))
print("a == c: {}".format(np.array_equal(a, c)))

print("\n- immutable array:")
v = np.array([1, 2, 3])
v.flags.writeable = False
try:
    v[0] = 0
except ValueError:
    print("ValueError")

print("\n- cart to pol conversion:")

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return [rho, phi]

m = np.array([np.random.uniform(0, 8, 2).astype("int")
                for _ in range(10)])
print("cart:\n{}".format(m))
m = np.array(list(map(lambda t: cart2pol(t[0], t[1]), m)))
print("pol:\n{}".format(m))

print("\n- max suppression:")
v = np.array([4, 2, 4, 8, 3, 7])
print("initial: {}".format(v))
v[np.argmax(v)] = 0
print("changed: {}".format(v))

print("\n- mesh creation:")
x_space = np.round(np.arange(0, 1, 0.1), 2)
y_space = np.round(np.arange(0, 1, 0.1), 2)
# x_mesh, y_mesh = np.meshgrid(x, y)
# print("x_mesh:\n{}".format(x_mesh))
# print("y_mesh:\n{}".format(y_mesh))
mesh = []
print("mesh:")
for y in y_space:
    mesh.append([])
    for x in x_space:
        mesh[-1].append((x, y))
        print(mesh[-1][-1], end=" ")
    print()

print("\n- cauchy matrix:")
v1 = (np.random.rand(3) * 10).astype("int")
print("v1:\n{}".format(v1))
v2 = (np.random.rand(5) * 10).astype("int")
print("v2:\n{}".format(v2))
mc = np.zeros((len(v1), len(v2)))
for i in range(len(v1)):
    for j in range(len(v2)):
        mc[i, j] = 1 / (v1[i] - v2[j])
print("mc:\n{}".format(mc))

print("\n- most commonly used numpy types:")
print(np.iinfo(np.int64))
print(np.iinfo(np.uint64))
print(np.finfo(np.float64))
print(np.finfo(np.complex64))

print("\n- list printing:")
v = [1, 2, 3, 4, 5]
print("v:")
for ve in v:
    print(ve, end=" ")
print()

print("\n- closest element:")
vec = [1, 3, 6, 9, 8]
print("vec: {}".format(vec))
target = 5
delta_min = 1e9
for v in vec:
    delta = abs(v - target)
    if delta < delta_min:
        closest = v
        delta_min = delta
print("closest element for {} is {}".format(target, closest))

print("\n- image representation:")
img = np.zeros((6, 6, 3))
img[:, :2, :] = [1, 0, 0]
img[:, 2:4, :] = [0, 1, 0]
img[:, 4:6, :] = [0, 0, 1]
print(img.shape)
plt.imshow(img)
# plt.show()

print("\n- distance calculation:")

def dist(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

coords = np.array([np.random.uniform(-100, 100, 2).astype("int")
                for _ in range(5)])
print("coords:\n{}".format(coords))

dists = []
for i in range(len(coords) - 1):
    for j in range(i + 1, len(coords)):
        dists.append(dist(coords[i], coords[j]))
print("dists:\n{}".format(dists))
