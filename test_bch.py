import bchlib
import hashlib
import os
import random
from aux_functions import *

# create a bch object
import numpy as np

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
infoMessage(getLineNumber(), f'BCH_POLYNOMIAL = {BCH_POLYNOMIAL}')
infoMessage(getLineNumber(), f'BCH_BITS = {BCH_BITS}')
infoMessage(getLineNumber(), f'bch = {bch}')

# random data
data = bytearray(os.urandom(7))  # in our case - set to the len of secret + as many whitespaces as needed
infoMessage(getLineNumber(), f'generated random data = {data}')

# encode and make a "packet"
ecc = bch.encode(data)
packet = data + ecc
infoMessage(getLineNumber(), f'encoding and creating packet')
infoMessage(getLineNumber(), f'ecc (Error Correcting Code) = {ecc}')
infoMessage(getLineNumber(), f'packet = {packet}')

# print hash of packet
sha1_initial = hashlib.sha1(packet)
infoMessage(getLineNumber(), f' sha1_initial = {sha1_initial.hexdigest(),}') # print('sha1: %s' % (sha1_initial.hexdigest(),))

def bitflip(packet):
    byte_num = random.randint(0, len(packet) - 1)
    bit_num = random.randint(0, 7)
    packet[byte_num] ^= (1 << bit_num)

# make BCH_BITS errors
for _ in range(BCH_BITS):
    bitflip(packet)

# print hash of packet
sha1_corrupt = hashlib.sha1(packet)
infoMessage(getLineNumber(), f' sha1_corrupt = {(sha1_corrupt.hexdigest(),)}') # print('sha1: %s' % (sha1_corrupt.hexdigest(),))

initial_list = []
for char in sha1_initial.hexdigest():
    initial_list.append(char)

corrupt_list = []
for char in sha1_corrupt.hexdigest():
    corrupt_list.append(char)

differences = np.array(initial_list) == np.array(corrupt_list)
number_of_flipped_bits = np.count_nonzero(differences)
infoMessage(getLineNumber(), f'How many bits are different? {number_of_flipped_bits}')
# print(f'how much different?: {number_of_flipped_bits}')

# de-packetize
data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
infoMessage(getLineNumber(), f'De - packeting')
infoMessage(getLineNumber(), f'ecc (Error Correcting Code) = {ecc}')
infoMessage(getLineNumber(), f'data = {data}')
infoMessage(getLineNumber(), f'packet decoded = {packet.decode}')

# print(f'data: \n{data}\n\necc: \n{ecc}\n\npacket decoded: {packet.decode}\n')  # debug

# correct
bitflips = bch.decode_inplace(data, ecc)
infoMessage(getLineNumber(), f'number of bit flips = {bitflips}')# print('bitflips: %d' % (bitflips))

# packetize
packet = data + ecc
infoMessage(getLineNumber(), f'packeting again. packet = {packet}')

# print hash of packet
sha1_corrected = hashlib.sha1(packet)
# print('sha1: %s' % (sha1_corrected.hexdigest(),))
infoMessage(getLineNumber(), f'sha1_corrected = {(sha1_corrected.hexdigest(),)}')

if sha1_initial.digest() == sha1_corrected.digest():
    print('Corrected!')
else:
    print('Failed')
