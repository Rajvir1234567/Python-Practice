source = open("info.txt", "r")

content = source.read()

destination = open("copy.txt", "w")

destination.write(content)

source.close()
destination.close()

print("File copied successfully")