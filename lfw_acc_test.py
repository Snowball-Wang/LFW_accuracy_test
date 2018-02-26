import lmdb
import caffe_pb2
from collections import OrderedDict
from math import sqrt

# Load lmdb file and save 512-d vector in feature_dict
def getFeatureVector(lmdb_path, feature_dict):
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key = int(key)
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            feature_dict[key] = list(datum.float_data)
    return feature_dict

# Get the mean value of 512-d feature vector
def getMean(feature_list):
    sum = [0 for j in range(512)]
    i = 0
    while i < len(feature_list):
        feature_add = list(map(lambda x: x[0] + x[1], zip(feature_list[i], feature_list[i+1])))
        sum = list(map(lambda x: x[0] + x[1], zip(feature_add, sum)))
        i = i + 2
    mu = [i/len(feature_list) for i in sum]
    return mu

# Get the feature vector normalized and compute the similarity
def getNormalized(feature_list, mu):
    feature_norm = feature_list[:]
    for i in range(len(feature_list)):
        feature_norm[i] = list(map(lambda x: x[0] - x[1], zip(feature_list[i], mu)))
    i = 0
    similarity_list = []
    while i < 12000:
        vec1 = feature_norm[i]
        vec2 = feature_norm[i+1]
        result = getCosineSimilarity(vec1, vec2)
        similarity_list.append(result)
        i = i + 2
    return similarity_list

def dotProduct(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(vector):
    return sqrt(dotProduct(vector, vector))

def getCosineSimilarity(v1, v2):
    return dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))

# Exclude the test feature set to get the validation feature set
def getValFeatureSet(pos_feature, neg_feature,i):
    val_feature = pos_feature[:600*i] + pos_feature[600*(i+1):]
    val_feature.extend((neg_feature[:600*i] + neg_feature[600*(i+1):]))
    return val_feature

# Exclude the test similarity set to get the validation similarity set
def getValSimiSet(similarity_list, i):
    pos_similarity = similarity_list[:3000]
    neg_similarity = similarity_list[3000:6000]
    val_similarity = pos_similarity[:300*i] + pos_similarity[300*(i+1):]
    val_similarity.extend(neg_similarity[:300*i] + neg_similarity[300*(i+1):])
    return val_similarity

# Get the test similarity set
def getTestSimiSet(similarity_list, i):
    pos_similarity = similarity_list[:3000]
    neg_similarity = similarity_list[3000:6000]
    test_similarity = pos_similarity[300*i:300*(i+1)] 
    test_similarity.extend(neg_similarity[300*i:300*(i+1)])
    return test_similarity

# Split the similarity into two halves and count the number 0 in label_err
def getAccuracy(threshold_value, similarity_list):
    length = len(similarity_list)
    label_original = similarity_list[:]
    label_test = similarity_list[:]
    for i in range(length/2):
        label_original[i] = 1
    for i in range(length/2, length):
        label_original[i] = 0

    for i in range(length):
        if similarity_list[i] > threshold_value:
            label_test[i] = 1
        elif similarity_list[i] <= threshold_value:
            label_test[i] = 0
        else:
            print "Error!"

    label_err = list(map(lambda x: x[0] - x[1],zip(label_original, label_test)))
    accuracy = label_err.count(0) / float(length)
    return accuracy

# Set the increasing step of the threshold and get the best threshold
# with maximum accuracy
def getThreshold(threshold_value, val_list):
    accuracy_list = []
    while threshold_value < 1.0:
        accuracy = getAccuracy(threshold_value, val_list)
        accuracy_list.append(accuracy)
        threshold_value = threshold_value + 0.0001
    accuracy_max = max(accuracy_list)
    threshold = [i for i, v in enumerate(accuracy_list) if v==accuracy_max]
    threshold_avg = sum(threshold) / float(len(threshold))
    return threshold_avg*0.0001

def main():
    root_dir = '/home/snowball/evaluation/' # this is the root directory, change it for your own directory
    lmdb_path = root_dir + 'data/nomirror'
    feature_dict = OrderedDict()
    print "Loading lmdb file..."
    feature_dict = getFeatureVector(lmdb_path, feature_dict)
    feature_list = feature_dict.values()
    print "There are %d images totally..." %(len(feature_list))
    pos_feature = feature_list[:6000]
    neg_feature = feature_list[6000:120000]
    
    acc_list = []
    threshold_list = []
    for i in range(10):
        print "Extracting deep features from the %dth fold..." %i
        threshold_init = 0.0
        val_feature = getValFeatureSet(pos_feature, neg_feature, i)
        mu = getMean(val_feature)
        similarity_list = getNormalized(feature_list, mu)
        val_list = getValSimiSet(similarity_list, i)
        threshold = getThreshold(threshold_init, val_list)
        test_list = getTestSimiSet(similarity_list, i)
        accuracy = getAccuracy(threshold, test_list)
        print "threshold: %.4f" %threshold
        print "acc: %.2f%%" %(accuracy*100)
        threshold_list.append(threshold)
        acc_list.append(accuracy)

    accuracy_avg = sum(acc_list) / 10.0
    threshold_avg = sum(threshold_list) / 10.0

    print "The average accuracy is %.2f%%" %(accuracy_avg*100)
    print "The threshold is %.4f" %threshold_avg
        

if __name__ == '__main__':
    main()
