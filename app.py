import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox


def prepare_and_train():

    csv_file = "flowers_dataset.csv"
    flower_names = ["Rose", "Tulip", "Sunflower", "Lily", "Daisy", "Orchid", "Iris", "Lavender", "Marigold", "Hibiscus"]

    if not os.path.exists(csv_file):

        data = []
        for i, name in enumerate(flower_names):
            
            center = i + 2
            samples = np.random.normal(loc=center, scale=0.5, size=(100, 4))
            for s in samples:
                data.append(list(np.abs(s)) + [name])
        
        df = pd.DataFrame(data, columns=["Petal_Size", "Sepal_Width", "Stem_Length", "Leaf_Width", "Flower_Type"])
        df.to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    X = df.drop('Flower_Type', axis=1)
    y = df['Flower_Type']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, flower_names


class FlowerApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Mega Flower Predictor (CSV Edition)")
        self.root.geometry("450x700")
        self.root.configure(bg="#f4f4f9")

        self.model, self.scaler, self.target_names = prepare_and_train()

        tk.Label(root, text="Multi-Flower Classification", font=("Arial", 18, "bold"), bg="#f4f4f9", fg="#2c3e50").pack(pady=20)

        input_frame = tk.Frame(root, bg="#f4f4f9")
        input_frame.pack(pady=10)

        self.inputs = {}
        self.feature_list = ["Petal Size", "Sepal Width", "Stem Length", "Leaf Width"]
        self.csv_columns = ["Petal_Size", "Sepal_Width", "Stem_Length", "Leaf_Width"]

        for i, feature in enumerate(self.feature_list):

            row = tk.Frame(input_frame, bg="#f4f4f9")
            row.pack(fill="x", pady=8)

            tk.Label(row, text=f"{feature} (1-12):", width=15, anchor="w", font=("Arial", 10), bg="#f4f4f9").pack(side="left", padx=20)
            entry = tk.Entry(row, font=("Arial", 10))
            entry.pack(side="right", expand=True, fill="x", padx=20)
            self.inputs[self.csv_columns[i]] = entry

        predict_btn = tk.Button(root, text="Identify Flower", command=self.predict, 
                               bg="#3498db", fg="white", font=("Arial", 12, "bold"), padx=30, pady=12, bd=0)
        predict_btn.pack(pady=20)

        self.result_label = tk.Label(root, text="Result: ---", font=("Arial", 16, "bold"), bg="#f4f4f9", fg="#7f8c8d")
        self.result_label.pack(pady=10)

        tk.Label(root, text="Try values between 2 and 11!", font=("Arial", 9, "italic"), bg="#f4f4f9", fg="#e67e22").pack()
        
        tk.Label(root, text="Example: Rose (2,2,2,2) | Hibiscus (11,11,11,11)", font=("Arial", 8), bg="#f4f4f9", fg="#7f8c8d").pack(pady=5)

        tk.Label(root, text="Supported Flowers:", font=("Arial", 10, "underline"), bg="#f4f4f9").pack(pady=10)
        species_text = ", ".join(self.target_names)
        tk.Label(root, text=species_text, font=("Arial", 9), bg="#f4f4f9", fg="#34495e", wraplength=400).pack()


    def predict(self):

        try:

            user_values = [float(self.inputs[col].get()) for col in self.csv_columns]
            
            user_values_scaled = self.scaler.transform([user_values])

            prediction = self.model.predict(user_values_scaled)[0]

            self.result_label.config(text=f"Result: {prediction}", fg="#27ae60")
            messagebox.showinfo("Prediction Result", f"This flower is identified as a {prediction}!")

        except ValueError:

            messagebox.showerror("Input Error", "Please enter valid numbers for all flower features.")

        except Exception as e:

            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":

    root = tk.Tk()
    app = FlowerApp(root)
    root.mainloop()
