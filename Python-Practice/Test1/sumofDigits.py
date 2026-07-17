num = int(input("Enter a number: "))
num = abs(num) 
total = 0
while num > 0:
    digit = num % 10
    total += digit
    num //= 10
print(f"The sum of the digits is: {total}")