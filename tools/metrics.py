from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(predictions, labels, threshold):
    precision_list = []
    recall_list = []
    f1_list = []
    spt_list = []
    acc_list = []

    binary_predictions = (predictions > threshold).astype(int)
    for i in range(labels.shape[1]):
        pred_class_i = binary_predictions[:,i]
        label_class_i = labels[:,i]
        acc = accuracy_score(label_class_i, pred_class_i)
        acc_list.append(acc)

        precision, recall, f1_score, support = precision_recall_fscore_support(label_class_i, pred_class_i, zero_division="warn")
        precision_list.append(precision[0])
        recall_list.append(recall[0])
        f1_list.append(f1_score[0])
        spt_list.append(support[0])

    return acc_list, precision_list, recall_list, f1_list, spt_list

def show_metrics(acc, precision, recall, f1, num):
    name = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    print(">>>======Val/Test Metrics======<<<")
    for i in range(len(precision)):
        print("Class_Name:{}  Num:{}  |  Accuracy:{:.5f} Precision:{:.5f} Recall:{:.5f} F1:{:.5f}".format(
            name[i], num[i], acc[i], precision[i], recall[i], f1[i]))

    average_acc = sum(acc)/len(acc)
    average_f1 = sum(f1)/len(f1)
    print("***Average            |  Acc:{:.5f} F1:{:.5f}".format(average_acc,average_f1))
    print(">>>============================<<<")

    return average_acc, average_f1