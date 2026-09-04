import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() > 0:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Controller '{controller.get_name()}' initialized.")
else:
    print("No controller found.")
    exit()

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Controller Input Test")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Handle controller events
        elif event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed")
        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button {event.button} released")
        elif event.type == pygame.JOYAXISMOTION:
            # Add a small threshold to prevent "joystick drift"
            if abs(event.value) > 0.1:
                print(f"Axis {event.axis} motion: {event.value:>6.3f}")
        elif event.type == pygame.JOYHATMOTION:
            print(f"Hat {event.hat} motion: {event.value}")

    # Your game logic and drawing code would go here

    pygame.display.flip()

pygame.quit()
