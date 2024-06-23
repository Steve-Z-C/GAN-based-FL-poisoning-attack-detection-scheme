# 引入必要的库
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from scipy import interp
from itertools import cycle
from scipy import interp
from matplotlib.font_manager import FontProperties

# font = FontProperties(fname=r'C:\Users\Administrator\AppData\Local\Microsoft\Windows\Fonts\FangZhengShuSong-GBK-1.ttf', size=12)
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun'

plt.rcParams['font.sans-serif']=['SimSun']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['font.size'] = 6 # 调整字体大小
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

model_normal = load_model(r"./client model\normal_2\global_model(acc=0.9&loss=0.3).hdf5")
# model_random = load_model(r"E:\GAPA\fashion_mnist\models\client model\random\global_model.hdf5")
model_random = load_model(r"./client model\random\global_model(random).hdf5")
model_target = load_model(r"./client model\target_2\global_model(acc=0.85&loss=0.4).hdf5")
model_detect_random = load_model(r"./client model\detect(random attack)\roc_model.hdf5")
model_detect_target = load_model(r"./client model\detect(target attack)\global_model.hdf5")

# #Ideal test
x_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\x_test_0.npy")
y_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\y_test_0.npy")

# #Original test
# x_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\original_test_x.npy")
# y_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\original_test_y.npy")
# x_test = np.load(r"E:\GAPA\fashion_mnist\generator sample\generation test\original_test_x.npy")
# y_test = np.load(r"E:\GAPA\fashion_mnist\generator sample\generation test\original_test_y.npy")

# #Generation test
# x_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\generation_test_x.npy")
# y_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\Server\generation_test_y.npy")
x_test = np.load(r"E:\GAPA\fashion_mnist\generator sample\generation test_1(0.71)\generation_test_x.npy")
y_test = np.load(r"E:\GAPA\fashion_mnist\generator sample\generation test_1(0.71)\generation_test_y.npy")
# x_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\generation_test_x.npy")
# y_test = np.load(r"E:\GAPA\fashion_mnist\dataset\Test_data\generation_test_y.npy")
print(x_test.shape, y_test.shape)

# y_test = np.argmax(y_test, axis=1)

nb_classes = 10

#得到标签值
y_pred_1 = model_normal.predict(x_test)
y_pred_2 = model_random.predict(x_test)
y_pred_3 = model_target.predict(x_test)
y_pred_4 = model_detect_random.predict(x_test)
y_pred_5 = model_detect_target.predict(x_test)
# y_pred_1 = np.argmax(y_pred_1, axis=1)
# y_pred_2 = np.argmax(y_pred_2, axis=1)
# y_pred_3 = np.argmax(y_pred_3, axis=1)
# y_pred_4 = np.argmax(y_pred_4, axis=1)
# y_pred_5 = np.argmax(y_pred_5, axis=1)

# AUC of micro-average
fpr_1 = dict()
tpr_1 = dict()
roc_auc_1 = dict()
fpr_1["micro"], tpr_1["micro"], _ = roc_curve(y_test.ravel(), y_pred_1.ravel())
roc_auc_1["micro"] = auc(fpr_1["micro"], tpr_1["micro"])
print('AUC micro-average: {:.4f}'.format(roc_auc_1["micro"]))

fpr_2 = dict()
tpr_2 = dict()
roc_auc_2 = dict()
fpr_2["micro"], tpr_2["micro"], _ = roc_curve(y_test.ravel(), y_pred_2.ravel())
roc_auc_2["micro"] = auc(fpr_2["micro"], tpr_2["micro"])
print('AUC micro-average: {:.4f}'.format(roc_auc_2["micro"]))

fpr_3 = dict()
tpr_3 = dict()
roc_auc_3 = dict()
fpr_3["micro"], tpr_3["micro"], _ = roc_curve(y_test.ravel(), y_pred_3.ravel())
roc_auc_3["micro"] = auc(fpr_3["micro"], tpr_3["micro"])
print('AUC micro-average: {:.4f}'.format(roc_auc_3["micro"]))

fpr_4 = dict()
tpr_4 = dict()
roc_auc_4 = dict()
fpr_4["micro"], tpr_4["micro"], _ = roc_curve(y_test.ravel(), y_pred_4.ravel())
roc_auc_4["micro"] = auc(fpr_4["micro"], tpr_4["micro"])
print('AUC micro-average: {:.4f}'.format(roc_auc_4["micro"]))

fpr_5 = dict()
tpr_5 = dict()
roc_auc_5 = dict()
fpr_5["micro"], tpr_5["micro"], _ = roc_curve(y_test.ravel(), y_pred_5.ravel())
roc_auc_5["micro"] = auc(fpr_5["micro"], tpr_5["micro"])
print('AUC micro-average: {:.4f}'.format(roc_auc_5["micro"]))

lw = 1
fig = plt.figure()
ax = fig.add_subplot(111)

# plt.plot(fpr_1["micro"], tpr_1["micro"], color='#2d85f0', label='无攻击(AUC={0:0.4f})'
#                ''.format(roc_auc_1["micro"]), linewidth=1)
# plt.plot(fpr_2["micro"], tpr_2["micro"], color='#f4433c', label='随机攻击(AUC={0:0.4f})'
#                ''.format(roc_auc_2["micro"]), linewidth=1)
# plt.plot(fpr_3["micro"], tpr_3["micro"], color='#0aa858', label='有目标攻击(AUC={0:0.4f})'
#                ''.format(roc_auc_3["micro"]), linewidth=1)
# plt.plot(fpr_4["micro"], tpr_4["micro"], color='#9c27b0', label='检测并剔除随机攻击(AUC={0:0.4f})'
#                ''.format(roc_auc_4["micro"]), linewidth=1)
# plt.plot(fpr_5["micro"], tpr_5["micro"], color='#ffbc32', label='检测并剔除有目标攻击(AUC={0:0.4f})'
#                ''.format(roc_auc_5["micro"]), linewidth=1)
plt.plot(fpr_1["micro"], tpr_1["micro"], color='#2d85f0', label='无攻击', linewidth=1)
plt.plot(fpr_2["micro"], tpr_2["micro"], color='#f4433c', label='无目标攻击', linewidth=1)
plt.plot(fpr_3["micro"], tpr_3["micro"], color='#0aa858', label='有目标攻击', linewidth=1)
plt.plot(fpr_4["micro"], tpr_4["micro"], color='#9c27b0', label='检测并剔除无目标攻击', linewidth=1)
plt.plot(fpr_5["micro"], tpr_5["micro"], color='#ffbc32', label='检测并剔除有目标攻击', linewidth=1)


plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率', fontsize=14)
plt.ylabel('真阳性率', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", fontsize=12)
# plt.axis('tight')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.axis('off')
plt.margins(0,0)
# plt.subplots_adjust(top=0.9,bottom=0.1,left=0.1,right=0.9,hspace=0,wspace=0)
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig(r"C:\Users\Administrator\Desktop\毕设实验结果\5.9\f.png", dpi=100, bbox_inches='tight')
plt.show()


#micro：多分类　　
# weighted：不均衡数量的类来说，计算二分类metrics的平均
# precision_1 = precision_score(y_test, y_pred_1, average='micro')
# precision_2 = precision_score(y_test, y_pred_2, average='micro')
# precision_3 = precision_score(y_test, y_pred_3, average='micro')
# recall_1 = recall_score(y_test, y_pred_1, average='micro')
# recall_2 = recall_score(y_test, y_pred_2, average='micro')
# recall_3 = recall_score(y_test, y_pred_3, average='micro')
# f1_score_1 = f1_score(y_test, y_pred_1, average='weighted')
# f1_score_2 = f1_score(y_test, y_pred_2, average='weighted')
# f1_score_3 = f1_score(y_test, y_pred_3, average='weighted')
# accuracy_score_1 = accuracy_score(y_test, y_pred_1)
# accuracy_score_2 = accuracy_score(y_test, y_pred_2)
# accuracy_score_3 = accuracy_score(y_test, y_pred_3)

# print("Normal model:")
# print("Precision_score:",precision_1)
# print("Recall_score:",recall_1)
# print("F1_score:",f1_score_1)
# print("Accuracy_score:",accuracy_score_1)

# print("Random attack model:")
# print("Precision_score:",precision_2)
# print("Recall_score:",recall_2)
# print("F1_score:",f1_score_2)
# print("Accuracy_score:",accuracy_score_2)

# print("Target attack model:")
# print("Precision_score:",precision_3)
# print("Recall_score:",recall_3)
# print("F1_score:",f1_score_3)
# print("Accuracy_score:",accuracy_score_3)












'''
# Compute ROC curve and ROC area for each class
# fpr_1 = dict()
# tpr_1 = dict()
# roc_auc_1 = dict()
# for i in range(nb_classes):
#     fpr_1[i], tpr_1[i], _ = roc_curve(y_test[:, i], y_pred_1[:, i])
#     roc_auc_1[i] = auc(fpr_1[i], tpr_1[i])

# fpr_1["micro"], tpr_1["micro"], _ = roc_curve(y_test.ravel(), y_pred_1.ravel())
# roc_auc_1["micro"] = auc(fpr_1["micro"], tpr_1["micro"])


# fpr_2 = dict()
# tpr_2 = dict()
# roc_auc_2 = dict()
# for i in range(nb_classes):
#     fpr_2[i], tpr_2[i], _ = roc_curve(y_test[:, i], y_pred_2[:, i])
#     roc_auc_2[i] = auc(fpr_2[i], tpr_2[i])

# fpr_2["micro"], tpr_2["micro"], _ = roc_curve(y_test.ravel(), y_pred_2.ravel())
# roc_auc_2["micro"] = auc(fpr_2["micro"], tpr_2["micro"])

# fpr_3 = dict()
# tpr_3 = dict()
# roc_auc_3 = dict()
# for i in range(nb_classes):
#     fpr_3[i], tpr_3[i], _ = roc_curve(y_test[:, i], y_pred_3[:, i])
#     roc_auc_3[i] = auc(fpr_3[i], tpr_3[i])

# fpr_3["micro"], tpr_3["micro"], _ = roc_curve(y_test.ravel(), y_pred_3.ravel())
# roc_auc_3["micro"] = auc(fpr_3["micro"], tpr_3["micro"])

# fpr_4 = dict()
# tpr_4 = dict()
# roc_auc_4 = dict()
# for i in range(nb_classes):
#     fpr_4[i], tpr_4[i], _ = roc_curve(y_test[:, i], y_pred_4[:, i])
#     roc_auc_4[i] = auc(fpr_4[i], tpr_4[i])

# fpr_4["micro"], tpr_4["micro"], _ = roc_curve(y_test.ravel(), y_pred_4.ravel())
# roc_auc_4["micro"] = auc(fpr_4["micro"], tpr_4["micro"])
'''