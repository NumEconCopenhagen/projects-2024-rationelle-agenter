import numpy as np
import matplotlib.pyplot as plt
from ExchangeEconomy import ExchangeEconomyClass
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

%load_ext autoreload
%autoreload 2

model = ExchangeEconomyClass()

par = model.par

###### OPGAVE 1 ######
alpha = 1/3
beta = 2/3


# a. total endowment
w1bar = 1.0
w2bar = 1.0

# Define simple utility functions for illustration
def u_A(x_A1, x_A2):
    return x_A1**alpha + x_A2**(1-alpha)

def u_B(x_B1, x_B2):
    return x_B1**beta + x_B2**(1-beta)

# Hypothetical initial endowments (adjust as necessary)
w_A1, w_A2 = 0.8, 0.3  # Initial endowment for Agent A
w_B1, w_B2 = 1-w_A1, 1-w_A2  # Initial endowment for Agent B

# Calculate utility at initial endowments
u_A_initial = u_A(w_A1, w_A2)
u_B_initial = u_B(w_B1, w_B2)

# b. figure set up for Edgeworth Box
fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
ax_A = fig.add_subplot(1, 1, 1)

ax_A.set_xlabel("$x_1^A$")
ax_A.set_ylabel("$x_2^A$")

ax_A.set_xlim([0, 1])
ax_A.set_ylim([0, 1])

N = 75
x_A1_range = np.linspace(0, 1, N+1)
x_A2_range = np.linspace(0, 1, N+1)

# Plot feasible allocations satisfying utility conditions
for x_A1 in x_A1_range:
    for x_A2 in x_A2_range:
        x_B1 = 1 - x_A1
        x_B2 = 1 - x_A2
        
        # Check if both agents' utility conditions are met
        if u_A(x_A1, x_A2) >= u_A_initial and u_B(x_B1, x_B2) >= u_B_initial:
            ax_A.scatter(x_A1, x_A2, color='blue', alpha=0.5)

# Highlight initial endowments
            
# limits
ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

ax_A.set_xlim([-0.1, w1bar + 0.1])
ax_A.set_ylim([-0.1, w2bar + 0.1])    
ax_B.set_xlim([w1bar + 0.1, -0.1])
ax_B.set_ylim([w2bar + 0.1, -0.1])

ax_A.scatter(w_A1, w_A2, color='red', label='Initial Endowment (A and B)')
ax_A.legend()

plt.grid(True)
plt.show()

###### OPGAVE 2 ######

# Constants
N = 75
p1_values = np.array([0.5 + 2*i/N for i in range(int(N*(2.5-0.5)/2) + 1)])

# Hypothetical initial endowments and p2 as the price of good 2 (assuming it's fixed for simplification)
omega_A1, omega_A2 = 0.5, 0.5
omega_B1, omega_B2 = 0.5, 0.5
p2 = 1  # Assuming the price of good 2 is constant for simplification

# Function to calculate optimal quantities - placeholder for demonstration
# These should be derived based on utility maximization specific to your model
def calculate_optimal_quantities(p1, omega):
    # Placeholder: linear relation to prices as a simplistic assumption
    x1 = omega[0] * p1 / (p1 + p2)
    x2 = omega[1] * p2 / (p1 + p2)
    return x1, x2

# Calculate errors in market clearing conditions
epsilon1 = []
epsilon2 = []

for p1 in p1_values:
    xA1_star, xA2_star = calculate_optimal_quantities(p1, (omega_A1, omega_A2))
    xB1_star, xB2_star = calculate_optimal_quantities(p1, (omega_B1, omega_B2))
    
    epsilon1.append(xA1_star - omega_A1 + xB1_star - omega_B1)
    epsilon2.append(xA2_star - omega_A2 + xB2_star - omega_B2)

# Results
print("p1 values:", p1_values)
print("Epsilon1:", epsilon1)
print("Epsilon2:", epsilon2)


###### OPGAVE 3 ######
min_error = float('inf')
market_clearing_p1 = None

for p1, e1, e2 in zip(p1_values, epsilon1, epsilon2):
    total_error = abs(e1) + abs(e2)  # Total error as the sum of absolute errors
    if total_error < min_error:
        min_error = total_error
        market_clearing_p1 = p1

print("Market Clearing Price for Good 1:", market_clearing_p1)

###### OPGAVE 4 ###### HER ER DER FOR SURE EN FEJL

initial_utility_B = omega_B1**alpha * omega_B2**(1-alpha)  # Initial utility for B

# Price set P1
P1 = np.array([0.5 + 2*i/N for i in range(int(N*(2.5-0.5)/2) + 1)])

# Placeholder utility function for A, depending on B's consumption
def utility_AonB(xB1, xB2):
    # Assuming A's utility is based on what's left after B's consumption
    return (1 - xB1)**alpha * (1 - xB2)**(1-alpha)

# Placeholder function to determine B's optimal consumption based on price and endowment
# In reality, this would involve solving B's utility maximization given p1 and their budget constraint
def optimal_consumption_B(p1, omega_B1, omega_B2):
    # Placeholder: Simply returns some fraction of the endowments based on price
    # You need to replace this with the actual calculation
    xB1 = omega_B1 / 2  # Placeholder calculation
    xB2 = omega_B2 / 2  # Placeholder calculation
    return xB1, xB2

max_utility_A = -np.inf
optimal_p1 = None

# Iterate through P1 to find the price that maximizes A's utility
for p1 in P1:
    xB1, xB2 = optimal_consumption_B(p1, omega_B1, omega_B2)
    current_utility_A = utility_AonB(xB1, xB2)
    
    if current_utility_A > max_utility_A:
        max_utility_A = current_utility_A
        optimal_p1 = p1

print(f"4A: Optimal price p1: {optimal_p1}, Max Utility for A: {max_utility_A}")


def demand_B(p1, omega_B1, omega_B2):
    # Simplified demand calculation
    xB1_star = omega_B1 / p1  # Placeholder
    xB2_star = omega_B2  # Placeholder
    return xB1_star, xB2_star


def objective_p1_discrete(p1):
    xB1_star, xB2_star = demand_B(p1, omega_B1, omega_B2)
    return -u_A(1-xB1_star, 1-xB2_star)  # Minimize negative utility for maximization

# Objective function for 4b: Any positive price
def objective_p1_continuous(p1):
    if p1 <= 0:  # Ensure positive price
        return float('inf')
    xB1_star, xB2_star = demand_B(p1, omega_B1, omega_B2)
    return -u_A(1-xB1_star, 1-xB2_star)

# Example optimization for 4b
res = minimize_scalar(objective_p1_continuous, bounds=(0.01, 10), method='bounded')
print(f"Optimal price p1: {res.x}, Max Utility: {-res.fun}")


###### OPGAVE 5 ######
x_A1_choices = np.linspace(0, 1, N+1)
x_A2_choices = np.linspace(0, 1, N+1)

max_utility_A = -np.inf
optimal_allocation = (None, None)

for x_A1 in x_A1_choices:
    for x_A2 in x_A2_choices:
        x_B1, x_B2 = 1 - x_A1, 1 - x_A2
        utility_B = x_B1**beta * x_B2**(1-beta)
        
        # Ensure Agent B is not worse off
        if utility_B >= initial_utility_B:
            utility_A = x_A1**alpha * x_A2**(1-alpha)
            
            if utility_A > max_utility_A:
                max_utility_A = utility_A
                optimal_allocation = (x_A1, x_A2)

print(f"Optimal Allocation for A under 5a: x_A1 = {optimal_allocation[0]}, x_A2 = {optimal_allocation[1]}")
print(f"Max Utility for A under 5a: {max_utility_A}")



# Correct utility function for A
def utility_A(x):
    x_A1, x_A2 = x  # Unpack the decision variables from the array
    return -(x_A1**alpha * x_A2**(1-alpha))  # Return negative utility for minimization

# Correct utility function for B (for use in constraints)
def utility_B(x):
    x_B1, x_B2 = 1 - x[0], 1 - x[1]  # Calculate B's consumption based on A's decision
    return x_B1**beta * x_B2**(1-beta)

# Initial utility for B with initial endowments
initial_utility_B = utility_B([omega_B1, omega_B2])

# Constraint to ensure B is not worse off
constraints = ({
    'type': 'ineq',
    'fun': lambda x: utility_B(x) - initial_utility_B
})

# Optimization problem setup for solving 5b
result = minimize(
    fun=utility_A,  # Objective function to minimize (maximize A's utility)
    x0=[0.5, 0.5],  # Initial guess for the optimization algorithm
    bounds=((0, 1), (0, 1)),  # Bounds for A's allocation decisions
    constraints=constraints  # Constraint to ensure B's utility is at least as high as initial
)

# Results
if result.success:
    optimal_x_A1, optimal_x_A2 = result.x
    print(f"5B: Optimal allocation for A: x_A1 = {optimal_x_A1:.4f}, x_A2 = {optimal_x_A2:.4f}")
    print(f"5B: Resulting utility for A: {-result.fun:.4f}")  # Negate because we minimized the negative utility
else:
    print("5B: Optimization was unsuccessful. Check the settings and constraints.")


###### OPGAVE 6 ######
# Utility functions assuming Cobb-Douglas form
def utility_A(x1, x2, alpha=0.5):
    return x1**alpha * x2**(1-alpha)

def utility_B(x1, x2, beta=0.5):
    return x1**beta * x2**(1-beta)

# Objective function for the social planner's problem
def objective(x):
    xA1, xA2 = x
    xB1, xB2 = 1 - xA1, 1 - xA2
    return -(utility_A(xA1, xA2) + utility_B(xB1, xB2))  # Negative for minimization

# Initial guess and bounds
x0 = [0.5, 0.5]
bounds = [(0, 1), (0, 1)]

# Solve the optimization problem
result = minimize(objective, x0, bounds=bounds)

if result.success:
    optimal_xA1, optimal_xA2 = result.x
    optimal_xB1, optimal_xB2 = 1 - result.x[0], 1 - result.x[1]
    print(f"Optimal Allocation for A: xA1 = {optimal_xA1:.4f}, xA2 = {optimal_xA2:.4f}")
    print(f"Optimal Allocation for B: xB1 = {optimal_xB1:.4f}, xB2 = {optimal_xB2:.4f}")
else:
    print("Optimization was unsuccessful.")


###### OPGAVE 7 ######
# Seed for reproducibility
np.random.seed(42)

# Generate 50 random (ωA1, ωA2) pairs where ωA1, ωA2 ~ U(0,1)
ωA1 = np.random.uniform(0, 1, 50)
ωA2 = np.random.uniform(0, 1, 50)

# Plot the set W
plt.figure(figsize=(8, 6))
plt.scatter(ωA1, ωA2, color='blue', label='W elements')
plt.title('Set W with 50 Elements')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('$ω_{A1}$')
plt.ylabel('$ω_{A2}$')
plt.grid(True)
plt.legend()
plt.show()


###### OPGAVE 8 ######

# Assuming total endowments for goods 1 and 2 are 1
total_endowment_1 = total_endowment_2 = 1

# Plotting initial endowments in the Edgeworth box
plt.figure(figsize=(8, 6))
plt.scatter(ωA1, ωA2, color='red', label='Initial Endowments ($\omega_{A}$)')
plt.xlim(0, total_endowment_1)
plt.ylim(0, total_endowment_2)
plt.xlabel('$\omega_{A1}$ (Good 1)')
plt.ylabel('$\omega_{A2}$ (Good 2)')
plt.title('Initial Endowments in the Edgeworth Box')
plt.grid(True)
plt.legend()
plt.show()
