numbers = input("Enter numbers separated by spaces: ").split()
unique = []
for num in numbers:
    if num not in unique:
        unique.append(num)
print("List with unique elements :")
print(unique)