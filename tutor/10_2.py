# Условие
# Даны два списка чисел. Посчитайте, сколько чисел содержится одновременно как в первом списке, так и во втором.
# Примечание. Эту задачу на Питоне можно решить в одну строчку.

print("len(lst1 & lst2) = ",
        len(set(input("lst1 = ").split()).intersection(set(input("lst2 = ").split()))))

