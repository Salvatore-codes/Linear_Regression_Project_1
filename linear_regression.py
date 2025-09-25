import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [45, 50, 60, 95, 90, 75, 80, 58, 90, 95]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied']]
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
hours_to_predict = pd.DataFrame([[7.5]], columns=['Hours_Studied'])
predicted_score = model.predict(hours_to_predict)
print(f"Predicted score for 7.5 hours of study: {predicted_score[0]:.2f}")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
plt.scatter(X, y, color='pink', label='Actual Data')
plt.plot(X, model.predict(X), color='yellow', linewidth=2, label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression Model')
plt.legend()
plt.show()