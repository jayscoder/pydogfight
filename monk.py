#
# import pymunk               # Import pymunk..
#
# space = pymunk.Space()      # Create a Space which contain the simulation
# space.gravity = 0,-981      # Set its gravity
#
# body = pymunk.Body()        # Create a Body
# body.position = 50,100      # Set the position of the body
#
# poly = pymunk.Poly.create_box(body) # Create a box shape and attach to body
# poly.mass = 10              # Set the mass on the shape
# space.add(body, poly)       # Add both body and shape to the simulation
#
# print_options = pymunk.SpaceDebugDrawOptions() # For easy printing
#
# for _ in range(100):        # Run simulation 100 steps in total
#     space.step(0.02)        # Step the simulation one step forward
#     space.debug_draw(print_options) # Print the state of the simulation


# if __name__ == '__main__':
#     import os
#
#     for name in os.listdir('.'):
#         if name.endswith('.json') and name.startswith('calc_optimal'):
#             os.remove(name)

