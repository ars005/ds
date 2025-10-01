import pandas as pd
import numpy as np
#students Data
students=pd.DataFrame({
    "student_id":[1,2,3,4,5],
    "name":["alica","bob","Charlie","David","eva"],
    "course_code":["CS101","CS101","EE202","CS101","EE202"]
})
#course Data
course=pd.DataFrame({
    "course_code":["CS101","EE202"],
    "course_name":["Computer science","Electrical Engineering"]
})
#marks data (multiple subjects per student , repeated student_id)
marks=pd.DataFrame({
    "student_id":[1,1,2,2,3,3,4,5,5,],
    "subject":["Math","Python","Math","Java","Circuits","Math","Python","Math","Electronics"],
    "marks":[90,85,70,80,75,88,60,92,78]
})
#------------------------------------
#merge students and course data
#------------------------------------
students_info=pd.merge(students,course,how="left",on="course_code")
print("merged student info:")
print(students_info)
#------------------------------------
#merge students and marks data
#------------------------------------
students_marks=pd.merge(students_info,marks,how="left",on="student_id")
print("merged students marks:")
print(students_marks)
#------------------------------------
#mapping grade assignment
#------------------------------------
def map_grade(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"
#----------------------------------------
#assign grades\
#----------------------------------------
students_marks["grade"]=students_marks["marks"].map(map_grade)
print("added grade column:",students_marks)
#remove duplicate per student subject
students_marks_unique=students_marks.drop_duplicates(subset=["student_id","subject"])
print("uniques subject marks per student:",students_marks_unique)
#average marks per student
avg_marks_per_course=students_marks_unique.groupby('course_name')["marks"].mean().reset_index()
print("average marks per course:",avg_marks_per_course)

#filter student with grade A
a_grade=students_marks_unique[students_marks_unique["grade"]=="A"]
print("stuents with grade A",a_grade)