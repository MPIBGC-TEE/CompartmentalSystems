from mm import multimethod

@multimethod(int, int)
def foo(a, b):
    return a/b

@multimethod(float, float)
def foo(a, b):
    return a*b

@multimethod(str, str)
def foo(a, b):
    return "\\frac{"+a+"}{"+ b+ "}"


print(foo(2,2))
print(foo(2.0,2.0))
print(foo("2","2"))
