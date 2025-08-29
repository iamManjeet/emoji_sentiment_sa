from sklearn.metrics import classification_report, accuracy_score, f1_score

def compute_metrics(y_true, y_pred, labels=None, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4)
    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}
