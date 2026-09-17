#Section 1: Decorators
def my_decorator(func):

    def wrapper():
        print("Wrapper function executed.")
        func()
        print("Wrapper function executed.")
    return wrapper

def hello_world_normal():
    print("Hello World")

hello_world_normal()
print("\n")

@my_decorator
def hello_world():
    print("Hello World")

hello_world()

#Section 2: Property Decorators
"""Data Validation, Private/Public (Encapsulation)"""

class Person:
    def __init__(self,name,age):
        self.__name = name
        self.__age = age
#__. ile private(erisilemez) hale getirdik
    #getter
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self,value):
        if value == "john":
            print("john not accepted")
        else:
            self.__name = value

kerem = Person(name="Kerem", age=22)
print(kerem.name)

#Section 3:Static Method
class MathOperations:

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def divide(x, y):
        return x / y


print(MathOperations.add(10,20))
print(MathOperations.divide(10,20))

# Section 4: Class Method
# alternative constructor
class Pizza:

    total_pizzas = 0

    def __init__(self, ingredients):
        self.ingredients = ingredients
        Pizza.total_pizzas += 1

    @classmethod
    def margherita(cls):
        return cls(["peynir", "domates", "fesleğen"])

    @classmethod
    def pepperoni(cls):
        return cls(["peynir", "sucuk", "domates"])

    @classmethod
    def get_total_pizzas(cls):
        return cls.total_pizzas


pizza1 = Pizza.margherita()
print(pizza1.ingredients)
pizza2 = Pizza.pepperoni()
print(pizza2.ingredients)
print(Pizza.get_total_pizzas())

#Section 5: Abstract Method
from abc import ABC, abstractmethod


class Animal(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def make_sound(self):
        pass

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def sleep(self):
        pass


class Dog(Animal):

    def make_sound(self):
        print("hav hav")

    def move(self):
        print("köpek yürüdü")

    def sleep(self):
        print("köpek uyudu")


barley = Dog("Barley")
barley.move()

#overloading, overriding, final

#Section 6: Overloading
from typing import overload, Union


class Calculator:

    @overload
    def add(self, a: int, b: int) -> int:
        ...

    @overload
    def add(self, a: int, b: int, c: int) -> int:
        ...

    def add(self, a: int, b: int, c: int | None = None) -> int:
        if c is None:
            return a + b
        return a + b + c


    @overload
    def process(self, value: int) -> int:
        ...

    @overload
    def process(self, value: str) -> str:
        ...

    def process(self, value: Union[int, str]) -> Union[int,str]:
        if isinstance(value, int):
            return value * 2
        elif isinstance(value, str):
            return value.upper()
        else:
            raise ValueError("Value must be a string or an integer")

calculator = Calculator()
print(calculator.add(10, 20))

result = calculator.process("a")
print(result)

#Section 7: Overriding
from typing import final

class BaseGame:

    def start(self):
        print("Game Started.")

    @final
    def calculate_score(self,points:int) -> int:
        bonus = 100
        return points + bonus

class MyGame(BaseGame):

    def start(self):
       #override
       print("My Game Started")

game = MyGame()
game.start()