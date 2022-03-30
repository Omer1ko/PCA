import numpy as np


S = np.array([[1,-2,5,4],
             [3, 2,1,-5],
             [-10, 1,-4,6]])
St = S.T
A=St@S

e, v = np.linalg.eig(A)
p = np.argsort(np.abs(e))[::-1]  # descending order
v = v[:, p]
e = e[p]


def question7a():
    print("Question 7a: \n")
    print("By definition, the distortion is 'how much energy we throw away?'")
    print("The distortion is equal to the sum of the lowest d - k eigenvalues of A.")
    print("The distortion is: ",e[2]+e[3],"\n")

def question7b():
    print("Question 7b: \n")

    column_0 = v[:, 0]
    column_1 = v[:, 1]
    U = np.block([[column_0],[column_1]]).T
    Ut=U.T
    print("Ut matrix is: ",Ut,"\n")
    return Ut



def question7c(Ut):
    print("Question 7c: \n")
    U=Ut.T
    print("Original examples: ","\n", S ,"\n\n\n")
    reductioned= Ut@St
    print("After dimensionality reduction: ","\n", reductioned.T, "\n\n\n")
    restored = (U@reductioned).T
    print("The restored examples: ","\n", restored, "\n\n\n")
    print("By definition, the distortion is 'how much energy we throw away?'")
    print("As we learn in class, it can be calculated as the squared norm of the subtraction between the restored \nvectors to the original vectors")
    print("The distortion is: ", np.linalg.norm(S-restored)*np.linalg.norm(S-restored))



if __name__ == '__main__':
    question7a()
    Ut=question7b()
    question7c(Ut)