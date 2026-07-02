import numpy as np
a = np.array([10, 20, 30, 40, 50])
b = np.zeros(5)
c = np.empty(5)
d = np.arange(0,10,2)
e = np.identity(3)  
while True:
 option = input("Enter choice (a, b, c, d, e): ")
 match option:
     case 'a':
         print("Array a:", a)
     case 'b':
         print("Array b (zeros):", b)
     case 'c':
         print("Array c (empty):", c)
     case 'd':
         print("Array d (arange):", d)
     case 'e':
         print("Array e (identity):", e)
     case _:
         print("Invalid choice.")
 choice = input("Do you want to continue? (y/n): ")
 if choice.lower() != 'y':
     break