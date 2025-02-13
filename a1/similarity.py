# -------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: similarity.py
# SPECIFICATION: Create a document-term matrix and find the two documents that 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: honestly probably like 15 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)
         # print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here

docTermMatrix = []
# print(type(documents[0][1]))
for row in documents:
  # Turn the sentence into individual words
  words = row[1].split(" ")
  # Count the occurrences of the word
  # for the first time the word appears insert that into that specific dictionary
  termInstanceDictionary = {}

  for word in words:
    if word in termInstanceDictionary:
      termInstanceDictionary[word] += 1
    else:
      termInstanceDictionary[word] = 1
         
  # Now that we have the count for each unique word in the row let's add it to docTermMatrix
  # to show that for this specific row it has this number of words
  docTermMatrix.append(termInstanceDictionary)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

# Let's first start by having variable to store the first document and second document that have the cosine similarity
# Additionally, lets create a variable to hold the cosine similarity
  
result_first_document_index = -1
result_second_document_index = -1
biggest_cosine_similarity = -999

def turnIntoLists(dict_x, dict_y):
  first_doc = dict_x
  second_doc = dict_y

  # Essentially merge the two lists so we get a list of the combined keys
  # We'll use this to compare both dictionaries and add 0's in the specific
  # dictionary where that key does not exist
  master_list = list({**first_doc, **second_doc}.keys())
  
  for key in master_list:
    if key not in first_doc:
      first_doc[key] = 0
    if key not in second_doc:
      second_doc[key] = 0

  # Now that we added 0's to both lists where they are missing the word
  # we have to sort them so that when we put it into the cosine_similarity
  # functions both lists align 

  x = []
  y = []

  for key in sorted(master_list):
    x.append(first_doc[key])
    y.append(second_doc[key])

  return x, y

# Now let's create a nested loop that will try every possible combination of rows
for first_doc_index in range(len(docTermMatrix)):
  for second_doc_index in range(len(docTermMatrix)):
    if first_doc_index == second_doc_index:
      # let's not compare the same document against itself
      continue
    
    X, Y = turnIntoLists(docTermMatrix[first_doc_index], docTermMatrix[second_doc_index])

    generate_cos_similarity = cosine_similarity([X], [Y])

    if (generate_cos_similarity > biggest_cosine_similarity):
      # We add +1 because the Tid starts at 1 unlike the list which starts at 0
      result_first_document_index = first_doc_index + 1
      result_second_document_index = second_doc_index + 1
      biggest_cosine_similarity = generate_cos_similarity


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
      
print(f"The most similar documents are document {result_first_document_index} and document {result_second_document_index} with cosine similarity = {biggest_cosine_similarity}")

# ANSWER
# The most similar documents are document 346 and document 392 with cosine similarity = [[0.7990928]]