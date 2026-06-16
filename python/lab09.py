import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB().fit(Xtr, ytr)
pred = gnb.predict(Xte)

print(f"Accuracy: {accuracy_score(yte, pred)*100:.2f}%\n")
print(f"Classification Report:\n{classification_report(yte, pred, zero_division=1)}")
print(f"Confusion Matrix:\n{confusion_matrix(yte, pred)}\n")
print(f"Cross-validation Accuracy: {cross_val_score(gnb, X, y, cv=5).mean()*100:.2f}%")

fig, axes = plt.subplots(3, 5, figsize=(12, 8))

for a, img, t, p in zip(axes.ravel(), Xte, yte, pred):
    a.imshow(img.reshape(64, 64), cmap="gray")
    a.set_title(f"True:{t}, Pred:{p}")
    a.axis("off")

plt.show()
