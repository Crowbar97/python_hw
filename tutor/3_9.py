# Улитка ползет по вертикальному шесту высотой h метров, поднимаясь за день на a метров, а за ночь спускаясь на b метров. На какой день улитка доползет до вершины шеста?
# Программа получает на вход натуральные числа h, a, b.
# Программа должна вывести одно натуральное число. Гарантируется, что a>b.

import math

h = int(input("h = "))
a = int(input("a = "))
b = int(input("b = "))

h1 = h - a
# print(f"h1 = {h1}")

if h1 > 0: 
    day_count = math.floor(h1 / (a - b))
    # print("count =", day_count)

    if not h1 % (a - b):
        print(day_count + 1)
    else:
        print(day_count + 2)
else:
    print(1)
