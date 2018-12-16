def calculate_macro_f1_score(predictions, true_labels):
    true_positives = [0 for i in range(11)]
    false_positives = [0 for i in range(11)]
    false_negatives = [0 for i in range(11)]

    if len(predictions) != len(true_labels):
        print "bug in code, length of predictions should match length of true_labels"
        return None
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            true_positives[predictions[i]] += 1
        else:
            false_positives[predictions[i]] += 1
            false_negatives[true_labels[i]] += 1

    total_classes = 0
    total_f1 = 0
    for i in range(11):
        if true_positives[i]==0 and false_positives[i]==0:
            continue
        elif true_positives[i]==0 and false_negatives[i]==0:
            continue
        prec = true_positives[i]*1.0/(true_positives[i] + false_positives[i])
        recall = true_positives[i]*1.0/(true_positives[i]+false_negatives[i])
        f1=0
        if prec+recall != 0:
            f1 = 2*prec*recall/(prec+recall)
            total_classes += 1
            total_f1 += f1
    return total_f1*100.0/total_classes

def calculate_micro_f1_score(predictions, true_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if len(predictions) != len(true_labels):
        print "bug in code, length of predictions should match length of true_labels"
        return None
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
    prec = true_positives*1.0/(true_positives + false_positives)
    recall = true_positives*1.0/(true_positives+false_negatives)
    return 2*prec*recall*100.0/(prec+recall)
