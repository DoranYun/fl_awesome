import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  # display “ - ”

acc_1 = pd.read_csv('test_accuacy_B_10_E_1.csv').to_numpy().tolist()
acc_1.insert(0, [0])
acc_2 = pd.read_csv('test_accuacy_B_10_E_5.csv').to_numpy().tolist()
acc_2.insert(0, [0])
acc_3 = pd.read_csv('test_accuacy_B_10_E_20.csv').to_numpy().tolist()
acc_3.insert(0, [0])
acc_4 = pd.read_csv('test_accuacy_B_50_E_1.csv').to_numpy().tolist()
acc_4.insert(0, [0])
acc_5 = pd.read_csv('test_accuacy_B_50_E_5.csv').to_numpy().tolist()
acc_5.insert(0, [0])
acc_6 = pd.read_csv('test_accuacy_B_50_E_20.csv').to_numpy().tolist()
acc_6.insert(0, [0])


epoch = np.arange(0, 51)

# fig, ax = plt.subplots(figsize=(8,6))
# ax.plot(epoch, acc_1, linestyle='-', color='red', label = 'B=10 E=1')
# ax.plot(epoch, acc_2, linestyle='--', color='red', label = 'B=10 E=5')
# ax.plot(epoch, acc_3, linestyle=':', color='red', label = 'B=10 E=20')
# ax.plot(epoch, acc_4, linestyle='-', color='blue', label = 'B=50 E=1')
# ax.plot(epoch, acc_5, linestyle='--', color='blue', label = 'B=50 E=5')
# ax.plot(epoch, acc_6, linestyle=':', color='blue', label = 'B=50 E=10')
# ax.set_xlabel('Communication Rounds', fontsize=16)
# ax.set_ylabel('Test Accuracy', fontsize=16)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.legend(fontsize=14)
# plt.title('Accuracy of FedAvg with IID Dataset', fontsize=16)
# plt.tight_layout()
# plt.savefig('FedAvg_iid.png', dpi=720)
# plt.show()

acc_7 = pd.read_csv('test_accuacy_B_10_E_1_noniid.csv').to_numpy().tolist()
acc_7.insert(0, [0])
acc_8 = pd.read_csv('test_accuacy_B_10_E_5_noniid.csv').to_numpy().tolist()
acc_8.insert(0, [0])
acc_9 = pd.read_csv('test_accuacy_B_10_E_20_noniid.csv').to_numpy().tolist()
acc_9.insert(0, [0])
acc_10 = pd.read_csv('test_accuacy_B_50_E_1_noniid.csv').to_numpy().tolist()
acc_10.insert(0, [0])
acc_11 = pd.read_csv('test_accuacy_B_50_E_5_noniid.csv').to_numpy().tolist()
acc_11.insert(0, [0])
acc_12 = pd.read_csv('test_accuacy_B_50_E_20_noniid.csv').to_numpy().tolist()
acc_12.insert(0, [0])


# fig, ax = plt.subplots(figsize=(8,6))
# plt.plot(epoch, acc_7, linestyle='-', color='red', label = 'B=10 E=1')
# plt.plot(epoch, acc_8, linestyle='--', color='red', label = 'B=10 E=5')
# plt.plot(epoch, acc_9, linestyle=':', color='red', label = 'B=10 E=20')
# plt.plot(epoch, acc_10, linestyle='-', color='blue', label = 'B=50 E=1')
# plt.plot(epoch, acc_11, linestyle='--', color='blue', label = 'B=50 E=5')
# plt.plot(epoch, acc_12, linestyle=':', color='blue', label = 'B=50 E=10')
# ax.set_xlabel('Communication Rounds', fontsize=16)
# ax.set_ylabel('Test Accuracy', fontsize=16)
# plt.legend(fontsize=14)
# plt.title('Accuracy of FedAvg with Non-IID Dataset', fontsize=16)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.tight_layout()
# plt.savefig('FedAvg_noniid.png', dpi=720)
# plt.show()





FedProx_1 = pd.read_csv('FedProx_test_accuracy_B_10_E_1.csv').to_numpy().tolist()
FedProx_1.insert(0, [0])
FedProx_2 = pd.read_csv('FedProx_test_accuracy_B_10_E_5.csv').to_numpy().tolist()
FedProx_2.insert(0, [0])
FedProx_3 = pd.read_csv('FedProx_test_accuracy_B_10_E_20.csv').to_numpy().tolist()
FedProx_3.insert(0, [0])
FedProx_4 = pd.read_csv('FedProx_test_accuracy_B_50_E_1.csv').to_numpy().tolist()
FedProx_4.insert(0, [0])
FedProx_5 = pd.read_csv('FedProx_test_accuracy_B_50_E_5.csv').to_numpy().tolist()
FedProx_5.insert(0, [0])
FedProx_6 = pd.read_csv('FedProx_test_accuracy_B_50_E_20.csv').to_numpy().tolist()
FedProx_6.insert(0, [0])


# epoch = np.arange(0, 51)


# fig, ax = plt.subplots(figsize=(8,6))
# ax.plot(epoch, FedProx_1, linestyle='-', color='red', label = 'B=10 E=1')
# ax.plot(epoch, FedProx_2, linestyle='--', color='red', label = 'B=10 E=5')
# ax.plot(epoch, FedProx_3, linestyle=':', color='red', label = 'B=10 E=20')
# ax.plot(epoch, FedProx_4, linestyle='-', color='blue', label = 'B=50 E=1')
# ax.plot(epoch, FedProx_5, linestyle='--', color='blue', label = 'B=50 E=5')
# ax.plot(epoch, FedProx_6, linestyle=':', color='blue', label = 'B=50 E=10')
# ax.set_xlabel('Communication Rounds', fontsize=16)
# ax.set_ylabel('Test Accuracy', fontsize=16)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.title('Accuracy of FedProx with IID Dataset', fontsize=16)
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig('FedProx_iid.png', dpi=720)
# plt.show()




FedProx_7 = pd.read_csv('FedProx_noniid_test_accuracy_B_10_E_1.csv').to_numpy().tolist()
FedProx_7.insert(0, [0])
FedProx_8 = pd.read_csv('FedProx_noniid_test_accuracy_B_10_E_5.csv').to_numpy().tolist()
FedProx_8.insert(0, [0])
FedProx_9 = pd.read_csv('FedProx_noniid_test_accuracy_B_10_E_20.csv').to_numpy().tolist()
FedProx_9.insert(0, [0])
FedProx_10 = pd.read_csv('FedProx_noniid_test_accuracy_B_50_E_1.csv').to_numpy().tolist()
FedProx_10.insert(0, [0])
FedProx_11 = pd.read_csv('FedProx_noniid_test_accuracy_B_50_E_5.csv').to_numpy().tolist()
FedProx_11.insert(0, [0])
FedProx_12 = pd.read_csv('FedProx_noniid_test_accuracy_B_50_E_20.csv').to_numpy().tolist()
FedProx_12.insert(0, [0])


# epoch = np.arange(0, 51)


# fig, ax = plt.subplots(figsize=(8,6))
# ax.plot(epoch, FedProx_7, linestyle='-', color='red', label = 'B=10 E=1')
# ax.plot(epoch, FedProx_8, linestyle='--', color='red', label = 'B=10 E=5')
# ax.plot(epoch, FedProx_9, linestyle=':', color='red', label = 'B=10 E=20')
# ax.plot(epoch, FedProx_10, linestyle='-', color='blue', label = 'B=50 E=1')
# ax.plot(epoch, FedProx_11, linestyle='--', color='blue', label = 'B=50 E=5')
# ax.plot(epoch, FedProx_12, linestyle=':', color='blue', label = 'B=50 E=10')
# ax.set_xlabel('Communication Rounds', fontsize=16)
# ax.set_ylabel('Test Accuracy', fontsize=16)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.title('Accuracy of FedProx with Non-IID Dataset', fontsize=16)
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig('FedProx_noniid.png', dpi=720)
# plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,12))


axes[0, 0].plot(epoch, acc_1, linestyle='-', color='red', label = 'B=10 E=1')
axes[0, 0].plot(epoch, acc_2, linestyle='--', color='red', label = 'B=10 E=5')
axes[0, 0].plot(epoch, acc_3, linestyle=':', color='red', label = 'B=10 E=20')
axes[0, 0].plot(epoch, acc_4, linestyle='-', color='blue', label = 'B=50 E=1')
axes[0, 0].plot(epoch, acc_5, linestyle='--', color='blue', label = 'B=50 E=5')
axes[0, 0].plot(epoch, acc_6, linestyle=':', color='blue', label = 'B=50 E=20')
axes[0, 0].set_xlabel('Communication Rounds', fontsize=16)
axes[0, 0].set_ylabel('Test Accuracy', fontsize=16)
axes[0, 0].tick_params(axis='x', labelsize=16)
axes[0, 0].tick_params(axis='y', labelsize=16)
axes[0, 0].set_title('Accuracy of FedAvg with IID Dataset', fontsize=16)
axes[0, 0].legend(fontsize=16)

axes[1, 0].plot(epoch, acc_7, linestyle='-', color='red', label = 'B=10 E=1')
axes[1, 0].plot(epoch, acc_8, linestyle='--', color='red', label = 'B=10 E=5')
axes[1, 0].plot(epoch, acc_9, linestyle=':', color='red', label = 'B=10 E=20')
axes[1, 0].plot(epoch, acc_10, linestyle='-', color='blue', label = 'B=50 E=1')
axes[1, 0].plot(epoch, acc_11, linestyle='--', color='blue', label = 'B=50 E=5')
axes[1, 0].plot(epoch, acc_12, linestyle=':', color='blue', label = 'B=50 E=20')
axes[1, 0].set_xlabel('Communication Rounds', fontsize=16)
axes[1, 0].set_ylabel('Test Accuracy', fontsize=16)
axes[1, 0].tick_params(axis='x', labelsize=16)
axes[1, 0].tick_params(axis='y', labelsize=16)
axes[1, 0].set_title('Accuracy of FedAvg with Non-IID Dataset', fontsize=16)
axes[1, 0].legend(fontsize=16)

axes[0, 1].plot(epoch, FedProx_1, linestyle='-', color='red', label = 'B=10 E=1')
axes[0, 1].plot(epoch, FedProx_2, linestyle='--', color='red', label = 'B=10 E=5')
axes[0, 1].plot(epoch, FedProx_3, linestyle=':', color='red', label = 'B=10 E=20')
axes[0, 1].plot(epoch, FedProx_4, linestyle='-', color='blue', label = 'B=50 E=1')
axes[0, 1].plot(epoch, FedProx_5, linestyle='--', color='blue', label = 'B=50 E=5')
axes[0, 1].plot(epoch, FedProx_6, linestyle=':', color='blue', label = 'B=50 E=20')
axes[0, 1].set_xlabel('Communication Rounds', fontsize=16)
axes[0, 1].set_ylabel('Test Accuracy', fontsize=16)
axes[0, 1].tick_params(axis='x', labelsize=16)
axes[0, 1].tick_params(axis='y', labelsize=16)
axes[0, 1].set_title('Accuracy of FedProx with IID Dataset', fontsize=16)
axes[0, 1].legend(fontsize=16)

axes[1, 1].plot(epoch, FedProx_7, linestyle='-', color='red', label = 'B=10 E=1')
axes[1, 1].plot(epoch, FedProx_8, linestyle='--', color='red', label = 'B=10 E=5')
axes[1, 1].plot(epoch, FedProx_9, linestyle=':', color='red', label = 'B=10 E=20')
axes[1, 1].plot(epoch, FedProx_10, linestyle='-', color='blue', label = 'B=50 E=1')
axes[1, 1].plot(epoch, FedProx_11, linestyle='--', color='blue', label = 'B=50 E=5')
axes[1, 1].plot(epoch, FedProx_12, linestyle=':', color='blue', label = 'B=50 E=20')
axes[1, 1].set_xlabel('Communication Rounds', fontsize=16)
axes[1, 1].set_ylabel('Test Accuracy', fontsize=16)
axes[1, 1].tick_params(axis='x', labelsize=16)
axes[1, 1].tick_params(axis='y', labelsize=16)
axes[1, 1].set_title('Accuracy of FedProx with Non-IID Dataset', fontsize=16)
axes[1, 1].legend(fontsize=16)
plt.subplots_adjust(hspace=1)
plt.tight_layout()
plt.savefig('zonghe.png', dpi=1440)
plt.show()

















brier_scores_iid = pd.read_csv('brier_scores_test.csv').to_numpy().tolist()
brier_scores_np_iid = np.array(brier_scores_iid)

brier_scores_noniid = pd.read_csv('brier_scores_test_niid.csv').to_numpy().tolist()
brier_scores_np_noniid = np.array(brier_scores_noniid)

brier_scores_de_6000 = pd.read_csv('brier_scores_de_6000.csv').to_numpy().tolist()
brier_scores_np_de_6000 = np.array(brier_scores_de_6000)

Fedprox_iid_brier_score = pd.read_csv('Fedprox_iid_brier_score_revised.csv').to_numpy().tolist()
Fedprox_iid_np_brier_score = np.array(Fedprox_iid_brier_score)

Fedprox_noniid_brier_score = pd.read_csv('Fedprox_noniid_brier_score_revised.csv').to_numpy().tolist()
Fedprox_noniid_np_brier_score = np.array(Fedprox_noniid_brier_score)

x_scale = np.arange(0, 1000, 20)

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(x_scale, brier_scores_np_iid/10, color='slateblue', label = 'FedAvg IID')
plt.plot(x_scale, brier_scores_np_noniid/10, color='red', label = 'FedAvg NON-IID')
plt.plot(x_scale, brier_scores_np_de_6000/10, color='black',linestyle='--', label = 'Deep Ensemble IID')
plt.plot(x_scale, Fedprox_iid_np_brier_score/10, color='orange', label = 'FedProx IID')
plt.plot(x_scale, Fedprox_noniid_np_brier_score/10, color='green', label = 'FedProx NON-IID')
plt.yscale("log")
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Brier Score', fontsize=16)
plt.title('Brier Score for FedProx', fontsize=16)
plt.legend(fontsize=14)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.savefig('Brier_FedProx.png', dpi=720)
plt.show()