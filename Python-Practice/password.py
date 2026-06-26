correct_password = "secure123"
password = ""
while password != correct_password:
    password = input("Enter the password:")
    if password == correct_password:
        print("Access granted!")
    else:
        print("Access denied!")