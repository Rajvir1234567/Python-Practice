# Student Record Management System using Functions and File Handling

# Function to Add Student
def add_student():
    roll = input("Enter Roll Number: ")
    name = input("Enter Name: ")
    marks = input("Enter Marks: ")

    file = open("students.txt", "a")
    file.write(roll + "," + name + "," + marks + "\n")
    file.close()

    print("Student record added successfully")


# Function to View Students
def view_students():

    try:
        file = open("students.txt", "r")

        print("\nStudent Records:\n")

        for line in file:
            data = line.strip().split(",")

            print("Roll No:", data[0])
            print("Name:", data[1])
            print("Marks:", data[2])
            print()

        file.close()

    except FileNotFoundError:
        print("No records found")


# Function to Search Student
def search_student():

    roll_search = input("Enter Roll Number to search: ")

    found = False

    try:
        file = open("students.txt", "r")

        for line in file:
            data = line.strip().split(",")

            if data[0] == roll_search:

                print("\nStudent Found")
                print("Roll No:", data[0])
                print("Name:", data[1])
                print("Marks:", data[2])

                found = True
                break

        file.close()

        if found == False:
            print("Student not found")

    except FileNotFoundError:
        print("File does not exist")


# Function to Delete Student
def delete_student():

    roll_delete = input("Enter Roll Number to delete: ")

    found = False

    try:
        file = open("students.txt", "r")
        lines = file.readlines()
        file.close()

        file = open("students.txt", "w")

        for line in lines:
            data = line.strip().split(",")

            if data[0] != roll_delete:
                file.write(line)
            else:
                found = True

        file.close()

        if found:
            print("Student deleted successfully")
        else:
            print("Student not found")

    except FileNotFoundError:
        print("File does not exist")


# Main Program
while True:

    print("\n--- Student Record Management System ---")
    print("1. Add Student")
    print("2. View Students")
    print("3. Search Student")
    print("4. Delete Student")
    print("5. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        add_student()

    elif choice == "2":
        view_students()

    elif choice == "3":
        search_student()

    elif choice == "4":
        delete_student()

    elif choice == "5":
        print("Exiting program...")
        break

    else:
        print("Invalid choice")