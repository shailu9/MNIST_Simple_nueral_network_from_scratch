import tkinter as tk
import numpy as np
from PIL import Image,ImageDraw

from model import SimpleNN


class DigitApp:
    def __init__(self,model:SimpleNN, root:tk.Tk):
        self.model=model
        self.root = root
        self.root.title("MNIST Digit Tester")

        self.canvas_size =200
        self.canvas = tk.Canvas(
            root,
            width= self.canvas_size,
            height= self.canvas_size,
            bg="black"
        )
        self.text_variable_predicated = tk.StringVar()
        self.text_variable_predicated.set(f"Predicated digit:")
        self.label_predicted = tk.Label(
            root,
            textvariable = self.text_variable_predicated
        )
        self.label_predicted.pack()
        self.canvas.pack()

        # PIL image processing
        self.image = Image.new("L",(self.canvas_size,self.canvas_size),"black")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        tk.Button(root, text="Predict", command=self.predict).pack()
        tk.Button(root, text="Clear", command=self.clear).pack()

        self.result_label = tk.Label(root, text="Draw a digit", font=("Arial", 16))

    def paint(self, event):
        r = 8
        x1, y1 = event.x - r, event.y - r
        x2, y2 = event.x + r, event.y + r

        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill="black")
        self.result_label.config(text="Draw a digit")
        self.text_variable_predicated.set(f"Predicated digit:")
        self.label_predicted.update()

    def preprocess(self):
        img = self.image.resize((28, 28))

        img_array = np.array(img)
        img_array = img_array / 255.0

        img_array = img_array.reshape(1, 784)

        return img_array

    def predict(self):
        print(f"Predict was called")
        processed = self.preprocess()

        output = self.model.forward(processed)
        print(f"output shape : {output.shape}")
        print(f"output :{output}")
        digit = np.argmax(output)
        confidence = output[0][digit] * 100
        self.text_variable_predicated.set(f"Predicated digit:{digit}, Confidence % : {confidence:.2f}")
        self.label_predicted.update()
        print(f"Predicated digit : {digit} Confidence % : {confidence:.2f}")
        self.result_label.config(text=f"Prediction: {digit}")

#if __name__ == '__main__':
#    root = tk.Tk()
#    app = DigitApp(root)
#    root.mainloop()