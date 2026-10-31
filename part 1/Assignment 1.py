import sympy as sp

# Define symbols
x, y = sp.symbols('x y')
theta_l, theta_r = sp.symbols('theta_l theta_r')
theta_dot_l, theta_dot_r = sp.symbols('theta_dot_l theta_dot_r')
r, x0, y0 = sp.symbols('r x0 y0')

# Belt constraints
eq1 = sp.Eq((x - x0) + (y - y0), r * theta_l)
eq2 = sp.Eq((x - x0) - (y - y0), r * theta_r)

# Solve for x and y
solution = sp.solve([eq1, eq2], [x, y])
position = sp.Matrix([solution[x], solution[y]])

# Calculate Jacobian
J = position.jacobian([theta_l, theta_r])

# Calculate velocities
angular_velocity = sp.Matrix([theta_dot_l, theta_dot_r])
velocity = sp.simplify(J * angular_velocity)

print("Position:")
sp.pprint(position)

print("\nJacobian:")
sp.pprint(J)

print("\nVelocity:")
sp.pprint(velocity)
