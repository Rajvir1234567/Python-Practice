num = int(input("Enter a number: "))
if (num < 0):
     print("Negative numbers cannot be palindromes.")
else:
    original = num
    reverse = 0
    while num > 0: 
         digit = num % 10
         reverse = reverse * 10 + digit
         num //= 10
    if original == reverse:
         print(f"{original} is a palindrome.")
    else:
     print(f"{original} is not a palindrome.")
             