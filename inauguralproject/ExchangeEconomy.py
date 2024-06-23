from types import SimpleNamespace
from scipy.optimize import brentq
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # preferences
        par.alpha = 1/3
        par.beta = 2/3

        # endowments
        par.w1A = 0.8
        par.w2A = 0.3

        par.w1B = 1 - par.w1A   
        par.w2B = 1 - par.w2A   

        # numeraire 
        par.p2 = 1

    def utility_A(self,x_A1,x_A2):
        par=self.par
        return x_A1**par.alpha * x_A2**(1-par.alpha)

    def utility_B(self,x_B1,x_B2):
        par=self.par
        return x_B1**par.beta * x_B2**(1-par.beta)

    def demand_A(self,p1):
        par=self.par
        x1A = par.alpha*(p1*par.w1A+par.p2*par.w2A)/p1      
        x2A = (1-par.alpha)*(p1*par.w1A+par.p2*par.w2A)/par.p2
        return x1A, x2A   

    def demand_Arand(self,p1,w1A,w2A):
        par=self.par
        x1A = par.alpha*(p1*w1A+par.p2*w2A)/p1
        x2A = (1-par.alpha)*(p1*w1A+par.p2*w2A)/par.p2
        return x1A, x2A


    def demand_B(self,p1):
        par=self.par
        x1B = par.beta*(p1*par.w1B+par.p2*par.w2B)/p1           
        x2B = (1-par.beta)*(p1*par.w1B+par.p2*par.w2B)/par.p2  
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def calculate_optimal_quantities(p1, omega):
        par=self.par
        x1 = omega[0] * p1 / (p1 + par.p2)
        x2 = omega[1] * par.p2 / (p1 + par.p2)
        return x1, x2
    
    def excess_demand_good1(self, p1):
        par=self.par
        x1A, _ = self.demand_A(p1)
        x1B, _ = self.demand_B(p1)
        return (x1A + x1B) - (par.w1A + par.w1B)

    def find_equilibrium_price(self):
        p1_eq = brentq(self.excess_demand_good1, a=0.01, b=10)
        return p1_eq
    
    def find_equilibrium_allocation(self, w1A, w2A, alpha, beta, p2=1):
        # Calculate initial endowments for B
        w1B, w2B = 1 - w1A, 1 - w2A
    
        # Assuming a method to find the equilibrium price p1, e.g., through optimization
        # Placeholder for the actual equilibrium price finding method
        p1_eq = self.find_equilibrium_price(w1A, w2A, alpha, beta, p2)

        # Demand functions would use p1_eq to calculate demands at equilibrium
        # Placeholder for demand calculation
        x1A_star, x2A_star = self.demand_A(p1_eq, w1A, w2A, alpha, p2)
        x1B_star, x2B_star = self.demand_B(p1_eq, w1B, w2B, beta, p2)

        return (x1A_star, x2A_star), (x1B_star, x2B_star)
    
    def update_endowments(self, w1A, w2A):
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A

    def calculate_equilibrium_allocations_rand(self, w1A_values, w2A_values):
        equilibrium_allocations = []
        for w1A, w2A in zip(w1A_values, w2A_values):
            self.update_endowments(w1A, w2A)
            p1_eq = self.find_equilibrium_price()
            allocation_A = self.demand_A(p1_eq)  # Implement this method in the class
            allocation_B = self.demand_B(p1_eq)  # Implement this method in the class
            equilibrium_allocations.append((allocation_A, allocation_B))
        return equilibrium_allocations

    def objective_function(self, p1, alpha, w1A, w2A, p2):
        par=self.par
        p1 = p1[0]  
        x1A = par.alpha * (p1 * par.w1A + par.p2 * par.w2A) / p1
        x2A = (1 - par.alpha) * (p1 * par.w1A + par.p2 * par.w2A) / par.p2
        
        utility = -(x1A ** par.alpha * x2A ** (1 - par.alpha))
        return utility
    

    def objective_function2(self, p1, alpha, w1B, w2B, p2):
        par = self.par
        p1 = p1[0]  

        # Calculate demand for Agent B
        x1B, x2B = self.demand_B(p1)
        
        # Calculate leftovers for Agent A
        x1A = par.w1A + par.w1B - x1B
        x2A = par.w2A + par.w2B - x2B
        
        # Ensure non-negative consumption
        x1A = max(0, x1A)
        x2A = max(0, x2A)
        
        # Calculate utility for Agent A
        utility = -(x1A ** par.alpha * x2A ** (1 - par.alpha))
        return utility

    def objective_A(self, x):
        par=self.par
        x1A, x2A = x
        return -(x1A**par.alpha * x2A**(1-par.alpha))
    
    def aggregate_utility(self, x):
        par=self.par
        x1A, x2A = x
        x1B, x2B = 1 - x1A, 1 - x2A
        return -(x1A**par.alpha * x2A**(1-par.alpha) + x1B**par.beta * x2B**(1-par.beta))  
