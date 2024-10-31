import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 設定標題
st.title('退休金計算器')

# 輸入退休年齡及餘命年限
retirement_year = st.number_input('請輸入退休年齡(年)', min_value=60, max_value=65, value=60)
retirement_month = st.number_input('請輸入退休年齡（月）', min_value=0, max_value=12, value=11)
life_expectancy = st.number_input('請輸入餘命年限', min_value=65, max_value=100, value=78)


s = 60
e = 65
y = np.linspace(s,e,61)
sy = (35 *12 + 2)/12

Q2 = (( 45800 * (sy + (y-60)) * 0.0155 ) * (1-(5-(y-60))*(4/100)))

retirement_age = retirement_year + retirement_month/12

# 計算每月可領的退休金

monthly_pension = (( 45800 * (sy + (retirement_age-60)) * 0.0155 ) * (1-(5-(retirement_age-60))*(4/100)))
years_pension = monthly_pension * (life_expectancy - retirement_age)*12

# 顯示每月可領的退休金
st.write(f'每月可領的退休金: {monthly_pension:.0f} 元')
st.write(f'總共可領的退休金: {years_pension:.0f} 元')



plt.figure(figsize=(10, 6))
plt.plot(y, Q2 * (life_expectancy-y) *12)
plt.plot(retirement_age, years_pension,'.')
plt.xlabel('retirement_age')
plt.ylabel('total pension')
plt.title(str(life_expectancy ) + 'years old')
plt.legend()
plt.grid(True)

# 顯示圖表
st.pyplot(plt)
