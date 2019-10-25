import sys
import numpy


def matrix_multiplication(matrix1, matrix2):
    print("running multiplication...")
    # matrix1 and 2 are list of list

    # check the format
    if len(matrix1[0]) != len(matrix2):
        print("Error: This two matrix cannot apply multiplication!")
        sys.exit(1)

    # calculate the format of answer
    answer = [[]*len(matrix2[0])] * len(matrix1)
    print("The answer should be ", len(answer), " * ", len(matrix2[0]))
    # multiplication may need 3 loops ?

    # use numpy
    numpy_answer = numpy.matmul(matrix1, matrix2)
    print("Answer: \n", numpy_answer)


list1 = [
    [1,2,1,5],
    [0,3,0,4],
    [-1,-2,0,0]
]
list2 = [
    [1],
    [3],
    [2],
    [1]
]
matrix_multiplication(list1, list2)
