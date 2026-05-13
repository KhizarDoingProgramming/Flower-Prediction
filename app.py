import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox, ttk
import os


def prepare_and_train():
    csv_file = "flowers_dataset.csv"
    
    if not os.path.exists(csv_file):
        # If file is missing, we notify the user instead of making fake data
        messagebox.showerror("Error", f"Dataset file '{csv_file}' not found!\nPlease ensure the dataset is in the same directory.")
        return None, None, []

    df = pd.read_csv(csv_file)
    
    # Dynamically get species names from CSV
    flower_names = sorted(df['Flower_Type'].unique().tolist())
    
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
        self.root.title("Flower Predictor Pro")
        self.root.geometry("500x750")
        self.root.configure(bg="#1e1e2f") # Dark premium theme

        # Load model
        self.model, self.scaler, self.target_names = prepare_and_train()
        
        if self.model is None:
            self.root.destroy()
            return

        # Header
        header_frame = tk.Frame(root, bg="#2d2d44", height=100)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="🌸 Flower Classifier", font=("Helvetica", 24, "bold"), 
                 bg="#2d2d44", fg="#a5a5ff").pack(pady=20)

        # Main Content
        main_frame = tk.Frame(root, bg="#1e1e2f")
        main_frame.pack(expand=True, fill="both", padx=40, pady=20)

        tk.Label(main_frame, text="Enter Flower Measurements", font=("Helvetica", 12), 
                 bg="#1e1e2f", fg="#8888aa").pack(pady=(0, 20))

        self.inputs = {}
        self.csv_columns = ["Petal_Size", "Sepal_Width", "Stem_Length", "Leaf_Width"]
        self.display_names = ["Petal Size", "Sepal Width", "Stem Length", "Leaf Width"]

        for i, feature in enumerate(self.display_names):
            label_text = feature
            
            tk.Label(main_frame, text=label_text, font=("Helvetica", 10, "bold"), 
                     bg="#1e1e2f", fg="#ffffff").pack(anchor="w", pady=(10, 2))
            
            entry = tk.Entry(main_frame, font=("Helvetica", 12), bg="#2d2d44", fg="white", 
                             insertbackground="white", bd=0, highlightthickness=1)
            entry.config(highlightbackground="#444466", highlightcolor="#a5a5ff")
            entry.pack(fill="x", ipady=8, pady=(0, 10))
            self.inputs[self.csv_columns[i]] = entry

        # Predict Button
        self.predict_btn = tk.Button(main_frame, text="IDENTIFY SPECIES", command=self.predict, 
                                    bg="#6c5ce7", fg="white", font=("Helvetica", 12, "bold"), 
                                    activebackground="#5849be", activeforeground="white",
                                    cursor="hand2", bd=0, pady=15)
        self.predict_btn.pack(fill="x", pady=30)

        # Result Area
        self.result_card = tk.Frame(main_frame, bg="#2d2d44", padx=20, pady=20)
        self.result_card.pack(fill="x")

        self.result_label = tk.Label(self.result_card, text="Ready for Analysis", 
                                     font=("Helvetica", 14, "bold"), bg="#2d2d44", fg="#8888aa")
        self.result_label.pack()

        # Footer / Supported Species
        footer_frame = tk.Frame(root, bg="#1e1e2f", pady=20)
        footer_frame.pack(fill="x")
        
        tk.Label(footer_frame, text="Supported Species:", font=("Helvetica", 9, "bold"), 
                 bg="#1e1e2f", fg="#444466").pack()
        
        species_text = " • ".join(self.target_names)
        tk.Label(footer_frame, text=species_text, font=("Helvetica", 8), 
                 bg="#1e1e2f", fg="#666688", wraplength=400).pack(pady=5)


    def predict(self):

        try:

            user_values = [float(self.inputs[col].get()) for col in self.csv_columns]
            
            user_values_scaled = self.scaler.transform([user_values])

            prediction = self.model.predict(user_values_scaled)[0]
            
            self.result_label.config(text=f"Match Found: {prediction}", fg="#00ff88")
            self.result_card.config(highlightthickness=2, highlightbackground="#00ff88")
            messagebox.showinfo("Result", f"This flower is definitely a {prediction}! 🌸")

        except ValueError:

            messagebox.showerror("Input Error", "Please enter valid numbers for all flower features.")

        except Exception as e:

            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":

    root = tk.Tk()
    app = FlowerApp(root)
    root.mainloop()
