import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/features_temporal.csv")

# Label encode
feature_cols = [c for c in df.columns if c not in 
                ['person_id','video_id','segment_type','frame_id',
                 'timestamp_sec','fps','label','blink_start']]

X = df[feature_cols]
y = df['label']

model = XGBClassifier(scale_pos_weight=len(y[y==0])/len(y[y==1]), 
                      n_estimators=200, random_state=42)
model.fit(X, y)

# Feature importance
imp = pd.Series(model.feature_importances_, index=feature_cols)
imp.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(8,8))
plt.tight_layout()
plt.savefig("feature_importance.png")
print(imp.sort_values(ascending=False).head(20))