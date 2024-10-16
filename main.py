import numpy as np
import time
from google.colab import files
from bitarray import bitarray

uploaded = files.upload()
input_file = open("/content/Input_file.txt","r")

content = input_file.read()

complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}

DNA_encoding_rules = {1: {"00": "A", "01": "C", "10": "G","11": "T"},
                      2: {"00": "A", "01": "G", "10": "C","11": "T"},
                      3: {"00": "C", "01": "A", "10": "T","11": "G"},
                      4: {"00": "C", "01": "T", "10": "A","11": "G"},
                      5: {"00": "G", "01": "A", "10": "T","11": "C"},
                      6: {"00": "G", "01": "T", "10": "A","11": "C"},
                      7: {"00": "T", "01": "C", "10": "G","11": "A"},
                      8: {"00": "T", "01": "G", "10": "C","11": "A"}}

# A -> |A>
DNA_basis_rules = { 1: {"A": np.array([1.0,0.0,0.0,0.0]), "C": np.array([0.0,1.0,0.0,0.0]), "G": np.array([0.0,0.0,1.0,0.0]),"T": np.array([0.0,0.0,0.0,1.0])},
                    2: {"A": np.array([1.0,0.0,0.0,0.0]), "G": np.array([0.0,1.0,0.0,0.0]), "C": np.array([0.0,0.0,1.0,0.0]),"T": np.array([0.0,0.0,0.0,1.0])},
                    3: {"C": np.array([1.0,0.0,0.0,0.0]), "A": np.array([0.0,1.0,0.0,0.0]), "T": np.array([0.0,0.0,1.0,0.0]),"G": np.array([0.0,0.0,0.0,1.0])},
                    4: {"C": np.array([1.0,0.0,0.0,0.0]), "T": np.array([0.0,1.0,0.0,0.0]), "A": np.array([0.0,0.0,1.0,0.0]),"G": np.array([0.0,0.0,0.0,1.0])},
                    5: {"G": np.array([1.0,0.0,0.0,0.0]), "A": np.array([0.0,1.0,0.0,0.0]), "T": np.array([0.0,0.0,1.0,0.0]),"C": np.array([0.0,0.0,0.0,1.0])},
                    6: {"G": np.array([1.0,0.0,0.0,0.0]), "T": np.array([0.0,1.0,0.0,0.0]), "A": np.array([0.0,0.0,1.0,0.0]),"C": np.array([0.0,0.0,0.0,1.0])},
                    7: {"T": np.array([1.0,0.0,0.0,0.0]), "C": np.array([0.0,1.0,0.0,0.0]), "G": np.array([0.0,0.0,1.0,0.0]),"A": np.array([0.0,0.0,0.0,1.0])},
                    8: {"T": np.array([1.0,0.0,0.0,0.0]), "G": np.array([0.0,1.0,0.0,0.0]), "C": np.array([0.0,0.0,1.0,0.0]),"A": np.array([0.0,0.0,0.0,1.0])}}

DNA_decoding_rules = {1: {"A": "00", "C": "01", "G": "10","T": "11"},
                      2: {"A": "00", "G": "01", "C": "10","T": "11"},
                      3: {"C": "00", "A": "01", "T": "10","G": "11"},
                      4: {"C": "00", "T": "01", "A": "10","G": "11"},
                      5: {"G": "00", "A": "01", "T": "10","C": "11"},
                      6: {"G": "00", "T": "01", "A": "10","C": "11"},
                      7: {"T": "00", "C": "01", "G": "10","A": "11"},
                      8: {"T": "00", "G": "01", "C": "10","A": "11"}}


encoding_rule = int(input("Choose any encoding rule number (1 to 8): "))

#Creating Bell States
def DNA_Bell_State(DNA_Vector):
    H_gate = np.array([[1, 0, 0, 1],[0, 1, 1, 0],[1, 0, 0, -1],[0, 1, -1, 0]])
    CNOT_gate = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])

    Entanglement = np.array([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*H_gate)] for X_row in CNOT_gate])

    Bell_State = np.matmul(Entanglement,DNA_Vector)

    return Bell_State

#S-DNA-C operations
Unitary_Gates = { "I_gate": np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]),
                  "X_gate": np.array([[0, 0, 1, 0],[0, 0, 0, 1],[1, 0, 0, 0],[0, 1, 0, 0]]),
                  "Z_gate": np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0],[0, 0, 0, -1]]),
                  "Y_gate": np.array([[0, 0, 1, 0],[0, 0, 0, 1],[-1, 0, 0, 0],[0, -1, 0, 0]]),
                  "XZ_gate": np.array([[0, 0, -1, 0],[0, 0, 0, -1],[1, 0, 0, 0],[0, 1, 0, 0]]),
                  "YZ_gate": np.array([[0, 0, -1, 0],[0, 0, 0, -1],[-1, 0, 0, 0],[0, -1, 0, 0]])}

def S_DNA_C(DNA_Bell_State, DNA_Basis):

    Bell_State_Message = []

    # I_GATE
    if (DNA_Basis == least_frequent_DNA):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["I_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["I_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["I_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["I_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # Z_GATE
    elif (DNA_Basis == complement[least_frequent_DNA]):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["Z_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["Z_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["Z_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["Z_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # X_GATE (00 -> 01)
    if (DNA_Bell_State.tolist() == [1.0, 0.0, 0.0, 1.0]) & (DNA_encoding_rules[encoding_rule]['01'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # X_GATE (01 -> 00)
    if (DNA_Bell_State.tolist() == [0.0, 1.0, 1.0, 0.0]) & (DNA_encoding_rules[encoding_rule]['00'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["X_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # Y_GATE (00 -> 10)
    if (DNA_Bell_State.tolist() == [1.0, 0.0, 0.0, 1.0]) & (DNA_encoding_rules[encoding_rule]['10'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # Y_GATE (01 -> 11)
    if (DNA_Bell_State.tolist() == [0.0, 1.0, 1.0, 0.0]) & (DNA_encoding_rules[encoding_rule]['11'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["Y_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # XZ_GATE (10 -> 00)
    if (DNA_Bell_State.tolist() == [0.0, 1.0, -1.0, 0.0]) & (DNA_encoding_rules[encoding_rule]['00'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # XZ_GATE (11 -> 01)
    if (DNA_Bell_State.tolist() == [1.0, 0.0, 0.0, -1.0]) & (DNA_encoding_rules[encoding_rule]['01'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["XZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # YZ_GATE (10 -> 11)
    if (DNA_Bell_State.tolist() == [0.0, 1.0, -1.0, 0.0]) & (DNA_encoding_rules[encoding_rule]['11'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    # YZ_GATE (11 -> 10)
    if (DNA_Bell_State.tolist() == [1.0, 0.0, 0.0, -1.0]) & (DNA_encoding_rules[encoding_rule]['10'] == DNA_Basis):
      if DNA_Basis == 'C':
        Bell_State_C = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_C)

      if DNA_Basis == 'T':
        Bell_State_T = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_T)

      if DNA_Basis == 'A':
        Bell_State_A = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_A)

      if DNA_Basis == 'G':
        Bell_State_G = np.matmul(Unitary_Gates["YZ_gate"],DNA_Bell_State).tolist()
        Bell_State_Message.append(Bell_State_G)

    return Bell_State_Message


tic = time.perf_counter()
#Text conversion to Binary
binary_str = ''.join(format(x, '08b') for x in bytearray(content, 'utf-8'))
binary_list = [binary_str[i: i+2] for i in range(0, len(binary_str), 2)]
#print("\nThe original string is :" + "\n" + content + "\n")
#print("The string after ASCII binary conversion is :" + "\n" + binary_str + "\n")

#Binary conversion to DNA
DNA_list = []
for num in binary_list:
    for key in list(DNA_encoding_rules[encoding_rule].keys()):
        if num == key:
          DNA_list.append(DNA_encoding_rules[encoding_rule].get(key))

DNA_str = "".join(DNA_list)

print("DNA strand:" + " " + DNA_str + "\n")

#Finding the least frequent DNA letter
def least_frequent_char(string):
    freq = {char: string.count(char) for char in set(string)}
    return list(freq.keys())[np.argmin(list(freq.values()))]

least_frequent_DNA = least_frequent_char(DNA_str)

print(least_frequent_DNA)

Bell_State_Dic = {}
Bell_State_Message = []

#Using the least frequent DNA letter to compute S-DNA-C operations for the DNA strand
for element in range(0, len(DNA_str)):
    if DNA_str[element] == 'C':
        Bell_State_Dic.update({"Bell_C": S_DNA_C(DNA_Bell_State(DNA_basis_rules[encoding_rule][least_frequent_DNA]), 'C')})
        Bell_State_Message.append(Bell_State_Dic.get("Bell_C"))
    if DNA_str[element] == 'T':
        Bell_State_Dic.update({"Bell_T": S_DNA_C(DNA_Bell_State(DNA_basis_rules[encoding_rule][least_frequent_DNA]), 'T')})
        Bell_State_Message.append(Bell_State_Dic.get("Bell_T"))
    if DNA_str[element] == 'A':
        Bell_State_Dic.update({"Bell_A": S_DNA_C(DNA_Bell_State(DNA_basis_rules[encoding_rule][least_frequent_DNA]), 'A')})
        Bell_State_Message.append(Bell_State_Dic.get("Bell_A"))
    if DNA_str[element] == 'G':
        Bell_State_Dic.update({"Bell_G": S_DNA_C(DNA_Bell_State(DNA_basis_rules[encoding_rule][least_frequent_DNA]), 'G')})
        Bell_State_Message.append(Bell_State_Dic.get("Bell_G"))

Coded_Message = []

#The Coded Bell States message to be sent to Bob over Quantum Channel
for element in range(0,len(Bell_State_Message)):
    for key, value in Bell_State_Dic.items():
      if Bell_State_Message[element] == value:
          Coded_Message.append(key)

toc = time.perf_counter()
elapsed_time = toc - tic

#noisy quantum channel
noise_effect = input("Choose noise effects (1. High Noise Type 1, 2. High Noise Type 2, 3. Complementary and 4. No change): ")

Noisy_Coded_Message = []

#Noise effects
match noise_effect:

  case '1':
    for element in range(0,len(Coded_Message)):
      if Coded_Message[element] == 'Bell_C':
        Noisy_Coded_Message.append('Bell_T')

      if Coded_Message[element] == 'Bell_G':
        Noisy_Coded_Message.append('Bell_A')

      if Coded_Message[element] == 'Bell_T':
        Noisy_Coded_Message.append('Bell_C')

      if Coded_Message[element] == 'Bell_A':
        Noisy_Coded_Message.append('Bell_G')
    print(Noisy_Coded_Message)

  case '2':
    for element in range(0,len(Coded_Message)):
      if Coded_Message[element] == 'Bell_C':
        Noisy_Coded_Message.append('Bell_A')

      if Coded_Message[element] == 'Bell_G':
        Noisy_Coded_Message.append('Bell_T')

      if Coded_Message[element] == 'Bell_T':
        Noisy_Coded_Message.append('Bell_G')

      if Coded_Message[element] == 'Bell_A':
        Noisy_Coded_Message.append('Bell_C')
    print(Noisy_Coded_Message)

  case '3':
    for element in range(0,len(Coded_Message)):
      if Coded_Message[element] == 'Bell_C':
        Noisy_Coded_Message.append('Bell_G')

      if Coded_Message[element] == 'Bell_G':
        Noisy_Coded_Message.append('Bell_C')

      if Coded_Message[element] == 'Bell_T':
        Noisy_Coded_Message.append('Bell_A')

      if Coded_Message[element] == 'Bell_A':
        Noisy_Coded_Message.append('Bell_T')
    print(Noisy_Coded_Message)

  case '4':
    print(Coded_Message)

#print(Bell_State_Dic)
#print(Bell_State_Message)
print("Encryption time:", elapsed_time)

#Decrytion phase


def H_DNA_Basis(Bell_State_Vector):
  CNOT_gate = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
  H_DNA_Basis = (np.array([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Bell_State_Vector)] for X_row in CNOT_gate]))

  H_gate = np.array([[1, 0, 0, 1],[0, 1, 1, 0],[1, 0, 0, -1],[0, 1, -1, 0]])/(2)

  return np.matmul(np.transpose(H_gate), H_DNA_Basis).reshape(1,4)

def noisy_least_frequent_char(string):
    freq = {char: string.count(char) for char in set(string)}
    return list(freq.keys())[np.argmin(list(freq.values()))]

Sense_strand = []

tic = time.perf_counter()

#No change
if noise_effect == '4':
  for element in range(0,len(Coded_Message)):
    for key, value in Bell_State_Dic.items():
      if Coded_Message[element] == key:
        if (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['C'].tolist()]:
          Sense_strand.append('C')

        elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['T'].tolist()]:
          Sense_strand.append('T')

        elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['A'].tolist()]:
          Sense_strand.append('A')

        elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['G'].tolist()]:
          Sense_strand.append('G')

#Noise Effects
elif (noise_effect == '1') | (noise_effect == '2') | (noise_effect == '3'):
  DNA_strand = []
  for element in range(0,len(Noisy_Coded_Message)):
      for key, value in Bell_State_Dic.items():
        if Noisy_Coded_Message[element] == key:
            if (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['C'].tolist()]:
                DNA_strand.append('C')

            elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['T'].tolist()]:
                DNA_strand.append('T')

            elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['A'].tolist()]:
                DNA_strand.append('A')

            elif (H_DNA_Basis(np.array(value).reshape(4,1))).tolist() == [DNA_basis_rules[encoding_rule]['G'].tolist()]:
                DNA_strand.append('G')

  Noisy_DNA_strand = ''.join([e for e in DNA_strand])
  print("Noisy DNA strand:" + " " + Noisy_DNA_strand + "\nNoise effect: " + noise_effect + "\n")

  print(noisy_least_frequent_char(Noisy_DNA_strand))


  match noise_effect:

  #High noise type 1
    case '1':
      if ((noisy_least_frequent_char(Noisy_DNA_strand) == 'T') & (least_frequent_DNA == 'C')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'C') & (least_frequent_DNA == 'T')):
        for element in Noisy_DNA_strand:
          if element == 'T':
            Sense_strand.append('C')

          if element == 'C':
            Sense_strand.append('T')

          if element == 'A':
            Sense_strand.append('G')

          if element == 'G':
            Sense_strand.append('A')

      elif ((noisy_least_frequent_char(Noisy_DNA_strand) == 'G') & (least_frequent_DNA == 'A')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'A') & (least_frequent_DNA == 'G')):
        for element in Noisy_DNA_strand:
          if element == 'G':
            Sense_strand.append('A')

          if element == 'A':
            Sense_strand.append('G')

          if element == 'C':
            Sense_strand.append('T')

          if element == 'T':
            Sense_strand.append('C')

    #High noise type 2
    case '2':
      if ((noisy_least_frequent_char(Noisy_DNA_strand) == 'A') & (least_frequent_DNA == 'C')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'C') & (least_frequent_DNA == 'A')):
        for element in Noisy_DNA_strand:
          if element == 'A':
            Sense_strand.append('C')

          if element == 'C':
            Sense_strand.append('A')

          if element == 'T':
            Sense_strand.append('G')

          if element == 'G':
            Sense_strand.append('T')

      elif ((noisy_least_frequent_char(Noisy_DNA_strand) == 'G') & (least_frequent_DNA == 'T')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'T') & (least_frequent_DNA == 'G')):
        for element in Noisy_DNA_strand:
          if element == 'G':
            Sense_strand.append('T')

          if element == 'T':
            Sense_strand.append('G')

          if element == 'C':
            Sense_strand.append('A')

          if element == 'A':
            Sense_strand.append('C')

    #Complementary noise
    case '3':
      if ((noisy_least_frequent_char(Noisy_DNA_strand) == 'G') & (least_frequent_DNA == 'C')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'C') & (least_frequent_DNA == 'G')):
        for element in Noisy_DNA_strand:
          if element == 'G':
            Sense_strand.append('C')

          if element == 'C':
            Sense_strand.append('G')

          if element == 'T':
            Sense_strand.append('A')

          if element == 'A':
            Sense_strand.append('T')

      elif ((noisy_least_frequent_char(Noisy_DNA_strand) == 'A') & (least_frequent_DNA == 'T')) | ((noisy_least_frequent_char(Noisy_DNA_strand) == 'T') & (least_frequent_DNA == 'A')):
        for element in Noisy_DNA_strand:
          if element == 'G':
            Sense_strand.append('C')

          if element == 'C':
            Sense_strand.append('G')

          if element == 'T':
            Sense_strand.append('A')

          if element == 'A':
            Sense_strand.append('T')


DNA_sense_strand = ''.join([e for e in Sense_strand])
print("DNA sense strand:" + " " + DNA_sense_strand + "\n")

binary_list = []

for num in DNA_sense_strand:
  for key in DNA_decoding_rules[encoding_rule].keys():
    if num == key:
        binary_list.append(DNA_decoding_rules[encoding_rule][key])

binary_str = "".join(binary_list)

bts = bitarray(binary_str)
ascs = bts.tobytes().decode('utf-8')

toc = time.perf_counter()
elapsed_time = toc - tic

f = open("Output_file.txt", "w")
print(ascs, file=f)
f.close()

#print("Alice's message:", ascs)
print("Decryption time:",elapsed_time)