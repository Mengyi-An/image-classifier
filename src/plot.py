import os

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# 创建一个客户端对象。括号里没传参数，所以它会去当前目录找 MLflow 默认的存储位置，也就是训练时生成的 mlruns
client = MlflowClient()
exp = client.get_experiment_by_name("image-classifier") # 按名字查找实验

# 按开始时间倒序取第 1 条 = 最近一次训练
run = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)[0]
# # 取出这一次 run 的 id
run_id = run.info.run_id 

# 把 val_loss和 val_acc 取出来返回成一个列表
val_loss = [m.value for m in client.get_metric_history(run_id, "val_loss")]
val_acc = [m.value for m in client.get_metric_history(run_id, "val_acc")]

train_loss = [m.value for m in client.get_metric_history(run_id, "train_loss")]
train_acc = [m.value for m in client.get_metric_history(run_id, "train_acc")]
epochs = range(1, len(val_loss) + 1)


plt.figure(figsize=(10, 4))

# plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, val_loss, label="val_loss")
plt.plot(epochs, train_loss, label="train_loss")
plt.legend(loc = "upper right")
plt.title("Training and Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, label="val_acc")
plt.plot(epochs, train_acc, label="train_acc")
plt.legend(loc = "upper right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acc")

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/epoch_curves.png", dpi=150, bbox_inches="tight")