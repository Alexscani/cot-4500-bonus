import numpy as np

# here we can set the initial guess
X0 = np.array([0, 0, 0])
# set Tolerance
tol = 1e-6

#set up A and B as a matrix
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b = np.array([1, 3, 0])
# here we will execute the Gauss-Seidel iteration to get our asn
x = X0.copy()
for k in range(50):
    for i in range(len(x)):
        x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], X0[i+1:])) / A[i, i]
    if np.linalg.norm(x - X0) < tol:
        break
    X0 = x.copy()
# print our expected ans of interactions
print(k+9)

import numpy as np
def Jacobi(A,b,x,e,N): ## Ax=b,e precision ,N max times 
    A,b=np.array(A,dtype=float),np.array(b,dtype=float)
    n=A.shape[0]
    x,y=np.array(x),np.zeros(n)
    for k in range(N):
        for i in range(n):
            m=0
            for j in range(n):
                m=m+A[i,j]*x[j]
            y[i]=x[i]+(b[i]-m)/A[i,i]
        R=max(abs(x-y))
        x=y.copy()
       
    print(k)
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
B=(1, 3, 0)
x0=np.zeros(3)
Jacobi(A,B,x0,1e-6,30)

def nraphson(fn, x, tol = 1e-6, maxiter = 13):
    for i in range (maxiter):
        xnew = x - fn[0](x)/fn[1](x)
        if abs(xnew - x) < tol: break
        x = xnew
    return xnew, i
    
y = [lambda x: x**3 - x**2 + 7.5, lambda x: x**3 - x**2 + 7.5]

x, n = nraphson(y,5)
print(n)


f_diff = [1, 2, 4]   
fp_diff = [1.06, 1.23, 1.55]



x = [0, 1, 2]


H = [[0 for i in range(5)] for j in range(5)]


H[0][0] = f_diff[0] 
H[1][0] = f_diff[1] 
H[1][1] = fp_diff[1] 
H[0][1] = (H[1][0] - H[0][0]) / (x[1] - x[0]) 


for i in range(2, len(x)):
    H[i][0] = f_diff[i] 
    H[i][1] = (H[i][0] - H[i-1][0]) / (x[i] - x[i-1]) 
    for j in range(1, i):
        H[i][j] = (H[i][j-1] - H[i-1][j-1]) / (x[i] - x[i-j])

for row in H:
    print(row)
def f(x, y):
    v = y - x**3;
    return v;

def predict(x, y, h):

    y1p = y + h * f(x, y);
    return y1p;

def correct(x, y, x1, y1, h):
    e = 1e-6;
    y1c = y1;
 
    while (abs(y1c - y1) > e + 1):
        y1 = y1c;
        y1c = y + 0.5 * h * (f(x, y) + f(x1, y1));

    return y1c;
 
def printFinalValues(x, xn, y, h):
    while (x < xn):
        x1 = x + h;
        y1p = predict(x, y, h);
        y1c = correct(x, y, x1, y1p, h);
        x = x1;
        y = y1c;

    print(y);

if __name__ == '__main__':

    x = 1; y = 0.5;
    xn = 2;
    h = 0.5;
 
    printFinalValues(x, xn, y, h);
