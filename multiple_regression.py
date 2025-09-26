import pandas as pd             
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression    
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Practice_Exams': [1, 2, 2, 3, 4, 4, 5, 5, 6, 7],
    'Class_Attendance': [8, 9, 8, 1, 9, 1, 1, 9, 10, 10],
    'Exam_Score': [55, 65, 70, 75, 80, 85, 90, 92, 95, 98]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Practice_Exams', 'Class_Attendance']] #Table from ehich the results are taken
y = df['Exam_Score'] #Table in which the results are shown
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data splitting complete.")
print(f"Training size: {len(X_train)} samples") #Samples from x
print(f"Testing size: {len(X_test)} samples") #Samples from x in y
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")
print(f"Intercept (Base Score): {model.intercept_:.2f}")
print("\nFeature Coefficients (How much a 1-unit change in the feature affects the score):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f} (Lower is better)")
print(f"R-squared (R²): {r2:.2f} (Closer to 1.0 is better)")
new_student = pd.DataFrame([[7.5, 4, 9]], 
                           columns=['Hours_Studied', 'Practice_Exams', 'Class_Attendance'])
predicted_score = model.predict(new_student)
print(f"\nPredicted score for a student with 7.5 hrs study, 4 exams, 9 classes: {predicted_score[0]:.2f}")
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='blue', label='Actual Data')
plt.plot(df['Hours_Studied'], model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Exam Score vs. Hours Studied (Model Trained on Multiple Features)')
plt.legend()
plt.grid(True)
plt.show()