import turtle
import random

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Colorful Star")
screen.setup(800, 600)

# Create a turtle for drawing
t = turtle.Turtle()
t.speed(0)  # Fastest speed

# List of vibrant colors
colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan", "magenta"]

# Function to draw a star with points
def draw_star(size, points):
    angle = 180 - (180 / points)
    
    for i in range(points):
        t.color(colors[i % len(colors)])  # Cycle through colors
        t.forward(size)
        t.right(angle)

# Function to draw a filled star
def draw_filled_star(size, points):
    t.begin_fill()
    draw_star(size, points)
    t.end_fill()

# Position the turtle
t.penup()
t.goto(0, 100)
t.pendown()

# Draw the main star
draw_filled_star(150, 5)

# Draw smaller surrounding stars
for i in range(8):
    t.penup()
    # Position stars in a circle around the main star
    angle = i * 45
    radius = 200
    x = radius * turtle.cos(turtle.radians(angle))
    y = radius * turtle.sin(turtle.radians(angle))
    t.goto(x, y)
    t.pendown()
    
    # Randomize size and points for each small star
    size = random.randint(20, 50)
    points = random.choice([5, 6, 7])
    
    # Random color for small stars
    t.color(random.choice(colors))
    t.begin_fill()
    draw_star(size, points)
    t.end_fill()

# Hide the turtle and display
t.hideturtle()

# Keep the window open
screen.exitonclick()