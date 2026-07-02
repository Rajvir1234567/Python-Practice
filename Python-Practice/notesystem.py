while True:
    print("\n--- Notes Management System ---")
    print("1. Write Notes")
    print("2. Append Notes")
    print("3. View Notes")
    print("4. Exit")

    choice = input("Enter your choice: ")

    # Write Option
    if choice == "1":
        note = input("Enter note: ")

        file = open("notes.txt", "w")
        file.write(note)

        file.close()

        print("Notes written successfully")

    # Append Option
    elif choice == "2":
        note = input("Enter note to append: ")

        file = open("notes.txt", "a")
        file.write("\n" + note)

        file.close()

        print("Notes appended successfully")

    # View Option
    elif choice == "3":
        file = open("notes.txt", "r")

        print("\nSaved Notes:")
        print(file.read())

        file.close()

    # Exit Option
    elif choice == "4":
        print("Exiting program...")
        break

    else:
        print("Invalid choice")