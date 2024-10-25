import tkinter as tk
from tkinter import messagebox
import os
from PIL import Image, ImageDraw
from Machine_Learning import model,np,plt,tf,cv2
def welcome_msg():
    messagebox.showinfo("Welcome","Welcome to the Handwritten Number Recognition App!")
root = tk.Tk()
root.withdraw()
welcome_msg()
root.deiconify()
root.title("Handwritten Number Recognition App")
root.geometry("300x350")

GRID_SIZE = 28
CELL_SIZE = 10
size = GRID_SIZE*CELL_SIZE
cells = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
rectangles = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
def toggle_cell(event):
    x, y = event.x // CELL_SIZE, event.y // CELL_SIZE  # Get grid position
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        cell = cells[y][x]
        cells[y][x] = 1 - cell  # Toggle cell state between 0 and 1
        new_color = "black" if cell == 0 else "white"
        canvas.itemconfig(rectangles[y][x], fill=new_color)
canvas = tk.Canvas(root, bg="white", width=size, height=size)

canvas = tk.Canvas(root, bg="white", width=size, height=size)
canvas.pack(pady=20)
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x1, y1 = j * CELL_SIZE, i * CELL_SIZE
        x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
        # Create rectangles and store them in the rectangles matrix
        rectangles[i][j] = canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
canvas.bind("<Button-1>", toggle_cell)
def savepng():
    image = Image.new('RGB',(size,size))
    draw = ImageDraw.Draw(image)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if cells[i][j]==1:
                x1,y1 = j*CELL_SIZE,i*CELL_SIZE
                x2,y2 = x1+ CELL_SIZE,y1+CELL_SIZE
                draw.rectangle([x1,y1,x2,y2],fill='black')
            else:
                x1,y1 = j*CELL_SIZE,i*CELL_SIZE
                x2,y2 = x1+ CELL_SIZE,y1+CELL_SIZE
                draw.rectangle([x1,y1,x2,y2],fill='white')
    image = image.resize((28,28))
    image.save("output.png")
save_button = tk.Button(root, text = "RECOGNIZE", command=savepng)
save_button.pack()

root.mainloop()
if os.path.isfile("output.png"):
    try:
        img = cv2.imread("output.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number {np.argmax(prediction)} appears")
        # argmax gixe the index of the field of highest probability
        # gives neuron with highest activation
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except: 
        print("Not clear!!")
# IT works!!