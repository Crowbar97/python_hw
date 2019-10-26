import numpy as np

# print("numpy version:", np.version.version)
# print("numpy config:")
# np.show_config()


print("[0] x 10:\n", np.zeros(10))
print("[1] x 10:\n", np.ones(10))
print("[2.5] x 10:\n", np.full(10, 2.5))

# help(np.add)

v = np.zeros(10)
v[4] = 1
print("custom:\n", v)

v = np.array([i for i in range(10, 50)])
print("[10 .. 49]:\n", v)

print("flipped:\n", np.flip(v))

print("3x3 0-8 matrix:\n", np.reshape(np.arange(0, 9), (3, 3)))

print("3x3x3 random:\n", (np.random.rand(3, 3, 3) * 10).astype("int"))

m = (np.random.rand(10, 10) * 10).astype("int")
print("10x10 random:\n", m)
print("min =", np.min(m))
print("max =", np.max(m))

v = (np.random.rand(30) * 10).astype("int")
print("30 random:\n", v)
print("mean = ", np.mean(v))

m = np.zeros((5, 5))
m[0] = 1
m[len(m) - 1] = 1
m[:, 0] = 1
m[:, len(m) - 1] = 1
print("bounds:\n", m)

print("0 * np.nan =", 0 * np.nan)
print("np.nan == np.nan =", np.nan == np.nan)
print("np.inf > np.nan =", np.inf > np.nan)
print("np.nan - np.nan =", np.nan - np.nan)

print("under diag:\n", np.diag([1, 2, 3, 4], -1))

# chess desk
cd = np.zeros((8, 8)).astype("int")
for i in range(len(cd)):
    for j in range(len(cd[i])):
        if (i + j) % 2:
            cd[i][j] = 1
print("chess desk:\n", cd)

pattern = np.array([[0, 1],
                    [1, 0]])
print("chess desk (tiling):\n", np.tile(pattern, (4, 4)))

n1, n2, n3 = 6, 7, 8
print("(n1, n2, n3) = (%d, %d, %d)" % (n1, n2, n3))

n = 100
print("n = %d" % n)

k = n // (n1 * n2)
j = (n % (n1 * n2)) // n2
i = (n % (n1 * n2)) % n2
print("(i, j, k) = (%d, %d, %d)" % (i, j, k))


