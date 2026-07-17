num = int(input("Enter a number: "))
if num < 0:
    print("Please enter a non-negative integer.")
else:
    original = num
    digits = len(str(num))
    total = 0
    while num > 0:
        digit = num % 10
        total += digit ** digits
        num //= 10
    if total == original:
        print(f"{original} is an Armstrong number.")
    else:
        print(f"{original} is not an Armstrong number.")