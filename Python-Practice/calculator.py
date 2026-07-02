result = float(input("Enter first operand:"))
while True:
    operator = input("Enter operator (+, -, *, /): ")
    operand = float(input("Enter next operand:"))
    if operator == '+':
        result += operand
    elif operator == '-':
        result -= operand
    elif operator == '*':
        result *= operand
    elif operator == '/':
        if operand == 0:
            print("Error: Division by zero is not allowed.")
        else:
            result /= operand
    else:
        print("Invalid operator. Please try again.")
        continue
    print("Current result:", result)
    choice = input("Do you want to continue? (y/n): ")
    if choice.lower() != 'y':
        break
print("Final result:", result)