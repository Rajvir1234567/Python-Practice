weight = float(input("Enter weight in kg: "))
height = float(input("Enter height in meters: "))

BMI = weight / (height ** 2)

print("BMI =", BMI)

if BMI < 18.5:
    print("Underweight")
elif BMI < 25:
    print("Normal weight")
elif BMI < 30:
    print("Overweight")
else:
    print("Obese")