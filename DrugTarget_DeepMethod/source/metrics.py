import tensorflow as tf


def tf_confusion_metrics(predict, real):
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)

    ones_like_actuals = tf.ones_like(actuals)  # 维度和actuals一样的全1的张量
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),  # 实际值为P
                tf.equal(predictions, ones_like_predictions)  # 预测值也为P
            ),
            "float"
        )
    )

    tn = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),  # 实际值为N
                tf.equal(predictions, zeros_like_predictions)  # 预测值为N
            ),
            "float"
        )
    )

    fp = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),  # 实际值为N
                tf.equal(predictions, ones_like_predictions)  # 预测值为P
            ),
            "float"
        )
    )

    fn = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),  # 实际值为P
                tf.equal(predictions, zeros_like_predictions)  # 预测值为N
            ),
            "float"
        )
    )
    return tp, fn, fp, tn


def ACC(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    accuracy = tf.add(tp,tn)/tf.add(tf.add(tp,fp),tf.add(fn,tn))
    return accuracy


def Precision(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    precision = float(tp) / (float(tp) + float(fp))
    return precision


def Recall(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    recall = float(tp) / (float(tp) + float(fn))
    return recall


def F1_score(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    precision = float(tp) / (float(tp) + float(fp))
    recall = float(tp) / (float(tp) + float(fn))
    f1_score = (2 * (precision * recall)) / (precision + recall)
    return f1_score


def TPR(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    tpr = float(tp) / (float(tp) + float(fn))
    return tpr


def FPR(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)

    fpr = float(fp) / (float(fp) + float(tn))
    return fpr


def FNR(P, Y):
    tp, fn, fp, tn = tf_confusion_metrics(P, Y)
    fnr = float(fn) / (float(tp) + float(fn))
    return fnr
