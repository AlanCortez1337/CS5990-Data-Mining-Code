# Converting row into instance counts
# words = ["apple", "apple", "apple", "banana", "banana", "orange"]

# termInstanceDictionary = {}

# for word in words:
#     print(word)
#     if word in termInstanceDictionary:
#         termInstanceDictionary[word] += 1
#     else:
#         termInstanceDictionary[word] = 1

# print(termInstanceDictionary)

# converting two dictionaries into equal sized lists

d1 = {"a": 1, "b": 2, "e": 7, "f": 8}
d2 = {"a": 2, "c": 1, "d": 1}

# d1_keys = d1.keys()
# d2_keys = d2.keys()

# l1_size = len(d1_keys)
# l2_size = len(d2_keys)

# X = []
# Y = []

master_list = {**d1, **d2}.keys()
print("OG d1: ", d1)
print("OG d2: ", d2)

for key in master_list:
    if key not in d2:
        d2[key] = 0
    if key not in d1:
        d1[key] = 0

print("----------------------")
print("d1: ", d1)
print("d2: ", d2)

print("----------------------")
print("Sort")


sorted_d1 = []
sorted_d2 = []

for key in sorted(master_list):
    sorted_d1.append(d1[key])
    
    sorted_d2.append(d2[key])

print(sorted_d1)
print(sorted_d2)

from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity([sorted_d1], [sorted_d2])

print("----------------------")
print(cos_sim)