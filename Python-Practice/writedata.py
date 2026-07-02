name = input("Enter name: ")
age = input("Enter age: ")
city = input("Enter city: ")

file = open("info.txt", "w")

file.write("Name: " + name + "\n")
file.write("Age: " + age + "\n")
file.write("City: " + city)

file.close()

print("Data written successfully")