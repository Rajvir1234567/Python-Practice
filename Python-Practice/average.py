marks = {"Rajvir":90,"Jaswinder":90,"Bhavya":99}
total_marks = 0
for mark in marks.values():
    total_marks += mark
    average_marks = float(total_marks/len(marks))
print("The average of students marks is:", average_marks)