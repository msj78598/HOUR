# استيراد المكتبات
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
import streamlit as st

# تحميل البيانات
data = pd.read_csv('hour_modified.csv')

# استكشاف البيانات
st.title("تحليل بيانات خدمة مشاركة الدراجات")
st.subheader("التحليل الاستكشافي")
st.write("عرض مصفوفة الارتباط التلقائي للميزات")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# إضافة ميزة التدرج الزمني كميزة إضافية
data['trend'] = range(len(data))

# اختيار الميزات وعمل تقسيم للبيانات
X = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'trend']]
y = data['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج MLE (Ridge Regression)
model_mle = Ridge()
model_mle.fit(X_train, y_train)
pred_mle = model_mle.predict(X_test)

# تدريب نموذج MAP (Gradient Boosting Regressor)
model_map = GradientBoostingRegressor()
model_map.fit(X_train, y_train)
pred_map = model_map.predict(X_test)

# حساب أداء النموذجين
mae_mle = mean_absolute_error(y_test, pred_mle)
med_ae_mle = median_absolute_error(y_test, pred_mle)
r2_mle = r2_score(y_test, pred_mle)

mae_map = mean_absolute_error(y_test, pred_map)
med_ae_map = median_absolute_error(y_test, pred_map)
r2_map = r2_score(y_test, pred_map)

# عرض أداء النموذجين
st.subheader("تقرير أداء النماذج")
st.write(f"نموذج MLE: MAE: {mae_mle:.2f}, Median AE: {med_ae_mle:.2f}, R²: {r2_mle:.2f}")
st.write(f"نموذج MAP: MAE: {mae_map:.2f}, Median AE: {med_ae_map:.2f}, R²: {r2_map:.2f}")

# رسم الرسوم البيانية لنتائج التنبؤ
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(y_test, pred_mle, alpha=0.5)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[0].set_title("MLE - التوقع مقابل القيم الحقيقية")
ax[0].set_xlabel("القيم الحقيقية")
ax[0].set_ylabel("التوقعات")

ax[1].scatter(y_test, pred_map, alpha=0.5, color='orange')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[1].set_title("MAP - التوقع مقابل القيم الحقيقية")
ax[1].set_xlabel("القيم الحقيقية")
ax[1].set_ylabel("التوقعات")

st.pyplot(fig)

# واجهة المستخدم للتنبؤ بناءً على المدخلات
st.subheader("نموذج توقع عدد الدراجات المؤجرة")
season = st.selectbox("اختر الموسم:", [1, 2, 3, 4])
yr = st.selectbox("اختر السنة:", [0, 1])
mnth = st.slider("اختر الشهر:", 1, 12)
hr = st.slider("اختر الساعة:", 0, 23)
holiday = st.selectbox("هل اليوم عطلة؟", [0, 1])
weekday = st.slider("اختر يوم الأسبوع:", 0, 6)
weathersit = st.slider("اختر حالة الطقس:", 1, 4)
temp = st.slider("أدخل درجة الحرارة (°C):", -10.0, 40.0)
atemp = st.slider("أدخل درجة الحرارة المحسوسة (°C):", -10.0, 50.0)
hum = st.slider("أدخل نسبة الرطوبة (%):", 0, 100)
windspeed = st.slider("أدخل سرعة الرياح (كم/س):", 0, 67)
trend = len(data)  # القيمة الأخيرة من التدرج الزمني (للأعوام الحالية)

# إعداد البيانات للمدخلات
input_data = pd.DataFrame([[season, yr, mnth, hr, holiday, weekday, weathersit, temp, atemp, hum, windspeed, trend]], 
                          columns=X.columns)

# التنبؤ باستخدام النماذج
predicted_count_mle = model_mle.predict(input_data)[0]
predicted_count_map = model_map.predict(input_data)[0]

# عرض التوقعات
st.write(f"التوقع باستخدام نموذج MLE: {predicted_count_mle:.2f} دراجة")
st.write(f"التوقع باستخدام نموذج MAP: {predicted_count_map:.2f} دراجة")

# حفظ التوقعات والنماذج في مجلد المشروع
input_data.to_csv("C:/hour/التوقعات.csv", index=False)
pd.DataFrame({'Actual': y_test, 'MLE Predictions': pred_mle, 'MAP Predictions': pred_map}).to_csv("C:/hour/أداء_النماذج.csv", index=False)

import joblib
joblib.dump(model_mle, 'C:/hour/model_mle.pkl')
joblib.dump(model_map, 'C:/hour/model_map.pkl')

st.write("تم حفظ التوقعات والنماذج في مجلد المشروع.")
