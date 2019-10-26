# Условие
# В списке все элементы различны. Поменяйте местами минимальный и максимальный элемент этого списка.

lst = input("lst = ").split()

print("source:\t\t", lst)

max_ind = lst.index(max(lst))
min_ind = lst.index(min(lst))

lst[max_ind], lst[min_ind] = lst[min_ind], lst[max_ind]

print("modified:\t", lst)

