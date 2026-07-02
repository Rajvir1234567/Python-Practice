file = open("info.txt", "a")

new_data = input("Enter new line: ")

file.write("\n" + new_data)

file.close()

print("Data appended successfully")