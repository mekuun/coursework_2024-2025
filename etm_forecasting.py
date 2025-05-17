import pandas as pd, numpy as np, matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

device           = 'cuda'  
cutoff_year      = 2020
future_years     = 2
min_docs_sum     = 30
min_active_years = 4
top_n            = 10
rolling_window   = 2

df = pd.read_csv('file2_with_topic_titles.csv')

year_topic_counts = df.groupby(['year','topic_title']).size().unstack(fill_value=0).astype(float)
yt = year_topic_counts
mask = (yt.sum() >= min_docs_sum) & \
       (yt.astype(bool).sum() >= min_active_years)
yt = yt.loc[:, mask]
top_topics = yt.sum().sort_values(ascending=False).head(top_n).index
yt = yt[top_topics]
print('Тем после фильтрации:', yt.shape[1])

yt_frac = (yt.div(yt.sum(1), 0)
             .rolling(rolling_window, 1).mean())

# metrics
metrics = []
for tid in yt_frac.columns:
    ser = yt_frac[tid]
    tr = ser[ser.index <= cutoff_year]
    ts = ser[ser.index >  cutoff_year]
    if ts.empty: continue
    df_tr = pd.DataFrame({'ds': pd.to_datetime(tr.index, format='%Y'),
                          'y': np.log1p(tr.values)})
    m = Prophet(yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False).fit(df_tr)
    fut = pd.DataFrame({'ds': pd.to_datetime(ts.index, format='%Y')})
    y_pred = np.expm1(m.predict(fut)['yhat'].values)
    y_true = ts.values
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true + 1e-6, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    smape_val = smape(y_true, y_pred)
    metrics.append([tid, mae, mape, rmse, smape_val])
    print(f'Тема {tid}: MAE={mae:.3f}  MAPE={mape:.1f}%  RMSE={rmse:.3f}  SMAPE={smape_val:.1f}%')

if metrics:
    mdf = pd.DataFrame(metrics, columns=['topic','MAE','MAPE','RMSE','SMAPE'])
    print('\nСредние метрики:\n', mdf[['MAE','MAPE','RMSE','SMAPE']].mean().round(2))

# vis test
import matplotlib.cm as cm

plt.figure(figsize=(14, 8))
colors = plt.get_cmap('tab10') 

for i, tid in enumerate(yt_frac.columns):
    ser = yt_frac[tid].copy()
    pre = ser[ser.index <= cutoff_year]
    post_real = ser[ser.index > cutoff_year]
    df_tr = pd.DataFrame({'ds': pd.to_datetime(pre.index, format='%Y'),
                          'y' : np.log1p(pre.values)})
    m = Prophet(yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False).fit(df_tr)
    fut = pd.DataFrame({'ds': pd.to_datetime(post_real.index, format='%Y')})
    fc = m.predict(fut)
    post_pred = pd.Series(np.expm1(fc['yhat'].values), index=post_real.index)
    color = colors(i % 10)
    plt.plot(ser.index, ser.values * 100, label=f'Тема {tid}', color=color)
    if 2024 in post_pred.index:
        plt.scatter([2024], [post_pred.loc[2024] * 100],
                    color=color, marker='o', s=50, zorder=5)

plt.xlabel('Год')
plt.ylabel('Доля темы (%)')
plt.title('Популярность тем с прогнозом на 2024 (точки)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('topic_forecast_colored_dot_2024.png')
plt.show()

# predict
pred = {}
for tid in yt_frac.columns:
    ser = yt_frac[tid]
    df_p = pd.DataFrame({'ds': pd.to_datetime(ser.index, format='%Y'),
                         'y': np.log1p(ser.values)})
    m = Prophet(yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False).fit(df_p)
    fut = m.make_future_dataframe(future_years, freq='YS')
    fc = m.predict(fut)
    tail = pd.Series(np.expm1(fc.tail(future_years)['yhat'].values),
                     index=fc.tail(future_years)['ds'].dt.year)
    pred[tid] = tail

all_future = sorted({y for s in pred.values() for y in s.index})
yt_frac = yt_frac.reindex(yt_frac.index.union(all_future), fill_value=0)
for tid, s in pred.items():
    yt_frac.loc[s.index, tid] = s.values

# vis
plt.figure(figsize=(14, 8))
for tid in yt_frac.columns:
    plt.plot(yt_frac.index, yt_frac[tid] * 100, label=f'Тема {tid}')
plt.xlabel('Год')
plt.ylabel('Доля темы (%)')
plt.yscale('log')
plt.title('Популярность тем (прогноз на 2025-2026)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('topic_popularity.png')
plt.show()
