import pandas as pd
import random
import numpy as np
import time
import sys
import os
from sklearn import svm
from sklearn import tree


def read_data(fileName):
    print(fileName)
    test_data = pd.read_csv(fileName)
    test_data = test_data.to_numpy()

    return test_data


def baseline_A1(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    random_assign = random.sample(range(1, len(result)+1), len(result))

    metric = {}
    for i in range(len(result)):
        metric.update({random_assign[i]: result[i]})

    new_metric = []
    for i in sorted(metric.keys(), reverse=True):
        new_metric.append([i, i, metric[i]])

    APFD_result = APFD(new_metric)

    return APFD_result, new_metric


def baseline_A2(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    count_failure = 0
    for t in range(len(result)):
        if result[t] == "F":
            count_failure += 1

    for i in range(len(result)):
        if i < count_failure:
            metric.append([i, i, "F"])
        else:
            metric.append([i, i, "P"])

    APFD_result = APFD(metric)

    return APFD_result, metric


def history_B1(run_num, data):
    if run_num < 5:
        return 0

    cur_run = list(data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    for t in range(len(result)):
        count_consecutive = 0

        for i in range(run_num - 1, -1, -1):
            if data[i][t + 1] == "F":
                consecutive_pass = count_consecutive
                break
            else:
                count_consecutive += 1

            if count_consecutive == run_num:
                consecutive_pass = count_consecutive

        metric.append([t, consecutive_pass, result[t]])

    metric = sorted(metric, key=lambda x: x[1])

    # random rank the same level data
    prev_count = metric[0][1]
    prev_list = [metric[0]]
    result_metric = []

    for j in range(1, len(metric)):
        if metric[j][1] != prev_count:
            prev_count = metric[j][1]

            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

            prev_list = [metric[j]]
        else:
            prev_list.append(metric[j])

        if j == len(metric) - 1:
            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

    APFD_result = APFD(result_metric)

    return APFD_result, result_metric


def history_B2(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    for t in range(len(result)):
        count_failure = 0

        for i in range(run_num - 1, -1, -1):
            if test_data[i][t + 1] == "F":
                count_failure += 1

        metric.append([t, count_failure / (run_num), result[t]])

    metric = sorted(metric, key=lambda x: x[1], reverse=True)

    # random rank the same level data
    prev_count = metric[0][1]
    prev_list = [metric[0]]
    result_metric = []

    for j in range(1, len(metric)):
        if metric[j][1] != prev_count:
            prev_count = metric[j][1]

            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

            prev_list = [metric[j]]
        else:
            prev_list.append(metric[j])

        if j == len(metric) - 1:
            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

    APFD_result = APFD(result_metric)

    return APFD_result, result_metric


def history_B3(alpha, run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    for i in range(len(result)):
        if test_data[0][i + 1] == "F":
            P = 1
        else:
            P = 0

        for j in range(run_num):
            if test_data[j][i + 1] == "F":
                P = alpha * 1 + (1 - alpha) * P
            else:
                P = (1 - alpha) * P

        metric.append([i, P, result[i]])

    metric = sorted(metric, key=lambda x: x[1], reverse=True)

    # random rank the same level data
    prev_count = metric[0][1]
    prev_list = [metric[0]]
    result_metric = []

    for j in range(1, len(metric)):
        if metric[j][1] != prev_count:
            prev_count = metric[j][1]

            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

            prev_list = [metric[j]]
        else:
            prev_list.append(metric[j])

        if j == len(metric) - 1:
            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

    APFD_result = APFD(result_metric)

    return APFD_result, result_metric


def history_B4(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    for i in range(len(result)):
        P = 0

        for j in range(1, run_num + 1):
            if run_num - j == 1:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.7 * 1
                else:
                    P += 0.7 * 0
            elif run_num - j == 2:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.2 * 1
                else:
                    P += 0.2 * 0
            else:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.1 * 1
                else:
                    P += 0.1 * 0

        metric.append([i, P, result[i]])

    metric = sorted(metric, key=lambda x: x[1], reverse=True)

    # random rank the same level data
    prev_count = metric[0][1]
    prev_list = [metric[0]]
    result_metric = []

    for j in range(1, len(metric)):
        if metric[j][1] != prev_count:
            prev_count = metric[j][1]

            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

            prev_list = [metric[j]]
        else:
            prev_list.append(metric[j])

        if j == len(metric) - 1:
            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

    APFD_result = APFD(result_metric)

    return APFD_result, result_metric


def feedback_E1(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = [[i, 0, result[i]] for i in range(len(result))]
    metric[0][1] = 1
    updated_metric = []
    while True:
        updated_metric.append(metric[0])
        t_finished = metric[0][2]
        t_finished_index = metric[0][0]
        metric.pop(0)
        if len(metric) == 0:
            break
        else:
            metric = co_failure(run_num, metric, t_finished, t_finished_index, test_data)

    APFD_result = APFD(updated_metric)

    return APFD_result, updated_metric


def co_failure(run_num, metric, t_finished, t_finished_index, test_data):
    history = []
    for index in range(run_num):
        history.append(test_data[index][t_finished_index + 1])

    for i in range(len(metric)):
        total_count = 0
        co_fail = 0
        cur_history = []
        for index in range(run_num):
            cur_history.append(test_data[index][metric[i][0] + 1])

        for j in range(len(history)):
            if history[j] == t_finished:
                total_count += 1
                if cur_history[j] == "F":
                    co_fail += 1

        if total_count == 0:
            priority = metric[i][1] + (0 - 0.5)
        else:
            priority = metric[i][1] + (co_fail / total_count - 0.5)
        metric[i][1] = priority

    metric = sorted(metric, key=lambda x: x[1], reverse=True)

    if len(metric) == 1:
        return metric

    # random rank the same level data
    prev_count = metric[0][1]
    prev_list = [metric[0]]
    result_metric = []

    for j in range(1, len(metric)):
        if metric[j][1] != prev_count:
            prev_count = metric[j][1]

            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

            prev_list = [metric[j]]
        else:
            prev_list.append(metric[j])

        if j == len(metric) - 1:
            random.shuffle(prev_list)
            for a in prev_list:
                result_metric.append(a)

    return result_metric


# flipping
def feedback_E2(run_num, test_data):
    if run_num < 5:
        return

    cur_run = list(test_data[run_num])
    result = cur_run[1:len(cur_run)]

    metric = []
    for i in range(len(result)):
        P = 0

        for j in range(1, run_num + 1):
            if run_num - j == 1:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.7 * 1
                else:
                    P += 0.7 * 0
            elif run_num - j == 2:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.2 * 1
                else:
                    P += 0.2 * 0
            else:
                if test_data[j - 1][i + 1] == "F":
                    P += 0.1 * 1
                else:
                    P += 0.1 * 0

        metric.append([i, P, result[i]])

    metric = sorted(metric, key=lambda x: x[1], reverse=True)

    for i in range(len(metric)):
        metric[i][1] = 0

    update_metric = []
    while True:
        if metric[0][2] == "P":
            update_metric.append(metric[0])
            metric.pop(0)
        else:
            history = [test_data[i][metric[0][0] + 1] for i in range(run_num)]

            # remove first element in metric
            update_metric.append(metric[0])
            metric.pop(0)

            for i in range(len(metric)):
                cur_history = [test_data[row][metric[i][0] + 1] for row in range(run_num)]

                flipping = 0
                prev_history = history[0]
                prev_cur_history = cur_history[0]
                for index in range(1, len(history)):
                    if history[index] != prev_history:
                        if cur_history[index] != prev_cur_history:
                            flipping += 1
                            prev_history = history[index]
                            prev_cur_history = cur_history[index]
                        else:
                            prev_history = history[index]
                            prev_cur_history = cur_history[index]
                    else:
                        prev_history = history[index]
                        prev_cur_history = cur_history[index]

                metric[i][1] = max(flipping, metric[i][1])

            metric = sorted(metric, key=lambda x: x[1], reverse=True)

            # random rank the same level data
            if len(metric) != 0:
                prev_count = metric[0][1]
                prev_list = [metric[0]]
                result_metric = []

                for j in range(1, len(metric)):
                    if metric[j][1] != prev_count:
                        prev_count = metric[j][1]

                        random.shuffle(prev_list)
                        for a in prev_list:
                            result_metric.append(a)

                        prev_list = [metric[j]]
                    else:
                        prev_list.append(metric[j])

                    if j == len(metric) - 1:
                        random.shuffle(prev_list)
                        for a in prev_list:
                            result_metric.append(a)

                metric = result_metric

        if len(metric) == 0:
            break

    APFD_result = APFD(update_metric)

    return APFD_result, update_metric


# terminator F2
def Terminator_F2(testData, resultData, N1, N2):
    execution_rank = []
    label_result = np.array(["undetermine" for i in range(len(resultData))])

    while len(execution_rank) < len(resultData):
        L_R = np.where(label_result == "fail")[0]

        if len(L_R) >= 1:
            X_id, X_prob = Train(label_result, testData, N1, N2)
        else:
            cur_unlabeled = np.where(label_result == "undetermine")[0]
            X_id = np.random.choice(cur_unlabeled, size=1, replace=False)

        for x in X_id:
            execution_rank.append(x)
            label_result = Execute(x, label_result, resultData)

    metric = []
    for item in execution_rank:
        metric.append([item, 0, resultData[item]])

    APFD_result = APFD(metric)

    return APFD_result, metric


def dataPreprocess(data, run_num):
    testData = []
    resultData = []

    for i in range(1, len(data[0])):
        tempTest = []

        for j in range(run_num):
            if data[j][i] == "P":
                tempTest.append(1)
            else:
                tempTest.append(0)

        testData.append(tempTest)

    for k in range(1, len(data[0])):
        resultData.append(data[run_num][k])

    return np.array(testData), np.array(resultData)


def Execute(x, label_result, resultData):
    new_label_result = list(label_result)

    if resultData[x] == "P":
        new_label_result[x] = "pass"
    else:
        new_label_result[x] = "fail"

    return np.array(new_label_result)


def certain(clf, test_data, unlabeled, N1):
    pos_at = list(clf.classes_).index("fail")
    prob = clf.predict_proba(test_data[unlabeled])[:, pos_at]
    order = np.argsort(prob)[::-1]

    return np.array(unlabeled)[order[:N1]], np.array(unlabeled)[order]


def uncertain(clf, test_data, unlabeled, N1):
    pos_at = list(clf.classes_).index("fail")
    prob = clf.predict_proba(test_data[unlabeled])[:, pos_at]
    train_dist = clf.decision_function(test_data[unlabeled])
    order = np.argsort(np.abs(train_dist))[:N1]

    return np.array(unlabeled)[order], np.array(prob)[order]


def Train(label_result, test_data, N1, N2):
    clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    clf_pre = tree.DecisionTreeClassifier(class_weight='balanced')

    poses = np.where(label_result == "fail")[0]  # get executed fail cases
    negs = np.where(label_result == "pass")[0]  # get executed pass cases
    left = poses

    decayed = list(left) + list(negs)  # executed test cases (L)
    unlabeled = np.where(label_result == "undetermine")[0]  # get unlabeled test cases
    number_randomPoint = min(len(decayed), len(unlabeled))
    unlabeled_update = np.random.choice(unlabeled, size=number_randomPoint,
                                        replace=False)  # presume |L| points from E \ L
    labels = np.array([x if x != "undetermine" else "pass" for x in label_result])  # presume pass

    all_neg = list(negs) + list(unlabeled_update)
    sample = list(decayed) + list(unlabeled_update)

    clf_pre.fit(test_data[sample], labels[sample])

    # aggresive sampling
    if len(poses) >= N2:
        pos_at = list(clf_pre.classes_).index("fail")
        train_dist = clf_pre.predict_proba(test_data[all_neg])[:, pos_at]
        negs_sel = np.argsort(train_dist)[: len(left)]
        sample = list(left) + list(np.array(all_neg)[negs_sel])

    clf.fit(test_data[sample], labels[sample])  # train linear SVM

    # certainty sampling and uncertainty sampling
    if len(poses) >= N2:  # certainty sampling
        X_id, X_prob = certain(clf, test_data, unlabeled, N1)
    else:
        X_id, X_prob = uncertain(clf, test_data, unlabeled, N1)

    return X_id, X_prob


# APFD calculation
def APFD(metric):
    n = len(metric)
    m = 0
    for i in range(len(metric)):
        if metric[i][2] == "F":
            m += 1
    apfd = 0
    num = 0

    for i in range(len(metric)):
        num += 1

        if metric[i][2] == "F":
            apfd += num

    apfd = 1 - float(apfd)/n/m + 1/(2*n)

    return apfd


def main():
    fileName = sys.argv[1]
    filePath = 'data/' + fileName

    test_data = read_data(filePath)

    x = []
    y_A1 = []
    y_A2 = []
    y_B1 = []
    y_B2 = []
    y_B3 = []
    y_B4 = []
    y_E1 = []
    y_E2 = []
    y_F2 = []

    for i in range(5, len(test_data)):
        x.append(i)

    t0 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        A1_result, lastMetricA1 = baseline_A1(i, test_data)
        y_A1.append(A1_result)
    print("A1 done")

    t1 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        A2_result, lastMetricA2 = baseline_A2(i, test_data)
        y_A2.append(A2_result)
    print("A2 done")

    t2 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        B1_result, lastMetricB1 = history_B1(i, test_data)
        y_B1.append(B1_result)
    print("B1 done")

    t3 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        B2_result, lastMetricB2 = history_B2(i, test_data)
        y_B2.append(B2_result)
    print("B2 done")

    t4 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        B3_result, lastMetricB3 = history_B3(0.5, i, test_data)
        y_B3.append(B3_result)
    print("B3 done")

    t5 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        B4_result, lastMetricB4 = history_B4(i, test_data)
        y_B4.append(B4_result)
    print("B4 done")

    t6 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        E1_result, lastMetricE1 = feedback_E1(i, test_data)
        y_E1.append(E1_result)
    print("C1 done")

    t7 = time.time()
    for i in range(5, len(test_data)):
        print(i)
        E2_result, lastMetricE2 = feedback_E2(i, test_data)
        y_E2.append(E2_result)
    print("C2 done")

    t8 = time.time()

    N1 = 10
    N2 = 20
    for i in range(5, len(test_data)):
        print(i)
        testData, resultData = dataPreprocess(test_data, i)
        F2_result, lastMetricF2 = Terminator_F2(testData, resultData, N1, N2)
        y_F2.append(F2_result)
    print("D1 done")

    t9 = time.time()

    print("A1: " + str(round(t1 - t0, 3)))
    print("A2: " + str(round(t2 - t1, 3)))
    print("B1: " + str(round(t3 - t2, 3)))
    print("B2: " + str(round(t4 - t3, 3)))
    print("B3: " + str(round(t5 - t4, 3)))
    print("B4: " + str(round(t6 - t5, 3)))
    print("C1: " + str(round(t7 - t6, 3)))
    print("C2: " + str(round(t8 - t7, 3)))
    print("D1: " + str(round(t9 - t6, 3)))

    # write file
    with open("result.txt", 'w') as f:
        # write A1
        f.write("A1\n")
        outStr = ""
        for item in y_A1:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write A2
        f.write("A2\n")
        outStr = ""
        for item in y_A2:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write B1
        f.write("B1\n")
        outStr = ""
        for item in y_B1:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write B2
        f.write("B2\n")
        outStr = ""
        for item in y_B2:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write B3
        f.write("B3\n")
        outStr = ""
        for item in y_B3:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write B4
        f.write("B4\n")
        outStr = ""
        for item in y_B4:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write E1
        f.write("C1\n")
        outStr = ""
        for item in y_E1:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write E2
        f.write("C2\n")
        outStr = ""
        for item in y_E2:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")

        # write F2
        f.write("D1\n")
        outStr = ""
        for item in y_F2:
            if outStr == "":
                outStr = outStr + str(item)
            else:
                outStr = outStr + " " + str(item)
        f.write(outStr + "\n")
        f.write("\n")


if __name__ == "__main__":
    main()

