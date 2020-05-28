
def stat(actual, observed):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(observed)): 
        if actual[i]==observed[i]==1:
           TP += 1
        if observed[i]==1 and actual[i]!=observed[i]:
           FP += 1
        if actual[i]==observed[i]==0:
           TN += 1
        if observed[i]==0 and actual[i]!=observed[i]:
           FN += 1

    return (TP, FP, TN, FN)