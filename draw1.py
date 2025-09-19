import turtle
import random

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Colorful Geometric Spiral")

# Create a turtle
t = turtle.Turtle()
t.speed(0)  # Fastest speed

# Function to draw a hexagon
def draw_hexagon(size, color):
    t.color(color)
    t.begin_fill()
    for _ in range(6):
        t.forward(size)
        t.left(60)
    t.end_fill()

# Function to draw the spiral pattern
def draw_spiral_pattern():
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan"]
    size = 10
    
    for i in range(100):
        # Change color every 5 iterations
        color = colors[i % len(colors)]
        
        # Draw a hexagon
        draw_hexagon(size, color)
        
        # Rotate and increase size slightly
        t.left(10)
        size += 2
        
        # Move to a new position for the next hexagon
        t.penup()
        t.forward(5)
        t.pendown()

# Draw the pattern
draw_spiral_pattern()

# Hide the turtle and display the result
t.hideturtle()
screen.exitonclick()
