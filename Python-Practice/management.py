students = []

while True:

    print("\n===== Student Management System =====")
    print("1. Add Student")
    print("2. Display Students")
    print("3. Search Student")
    print("4. Delete Student")
    print("5. Unique Courses")
    print("6. Exit")

    choice = input("Enter your choice: ")

    # Add Student
    if choice == "1":

        name = input("Enter student name: ")
        roll = input("Enter roll number: ")
        course = input("Enter course: ")

        print("Enter marks of 3 subjects")
        m1 = int(input("Subject 1: "))
        m2 = int(input("Subject 2: "))
        m3 = int(input("Subject 3: "))

        # Tuple for marks
        marks = (m1, m2, m3)

        # Dictionary for student details
        student = {
            "Name": name,
            "Roll": roll,
            "Course": course,
            "Marks": marks
        }

        # Add student into list
        students.append(student)

        print("Student added successfully!")

    # Display Students
    elif choice == "2":

        if len(students) == 0:
            print("No student records found.")

        else:
            for s in students:

                print("\nStudent Details")
                print("Name:", s["Name"])
                print("Roll:", s["Roll"])
                print("Course:", s["Course"])
                print("Marks:", s["Marks"])

                average = sum(s["Marks"]) / len(s["Marks"])
                print("Average Marks:", average)

    # Search Student
    elif choice == "3":

        roll_search = input("Enter roll number to search: ")

        found = False

        for s in students:

            if s["Roll"] == roll_search:

                print("\nStudent Found")
                print("Name:", s["Name"])
                print("Course:", s["Course"])
                print("Marks:", s["Marks"])

                found = True
                break

        if found == False:
            print("Student not found.")

    # Delete Student
    elif choice == "4":

        roll_delete = input("Enter roll number to delete: ")

        found = False

        for s in students:

            if s["Roll"] == roll_delete:

                students.remove(s)

                print("Student deleted successfully!")
                found = True
                break

        if found == False:
            print("Student not found.")

    # Unique Courses using Set
    elif choice == "5":

        courses = set()

        for s in students:
            courses.add(s["Course"])

        print("Unique Courses:")
        print(courses)

    # Exit
    elif choice == "6":

        print("Exiting program...")
        break

    else:
        print("Invalid choice!")